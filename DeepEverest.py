import ctypes
import heapq
from timeit import default_timer as timer

import numpy as np

from utils import l2_dist, get_group_activations_from_layer, binary_search, \
    get_layer_result_for_input_batch, get_partition_id_by_input_id, get_input_ids_by_partition_id, _get_double_pointers, \
    load_pickle, persist_index


def construct_index(index_lib, n_inputs, ratio, n_partitions, bits_per_input, layer_result):
    idx_idx = dict()
    cnt = 0

    for neuron_idx, _ in np.ndenumerate(layer_result[0]):
        idx_idx[neuron_idx] = cnt
        cnt += 1

    parameters = np.moveaxis(layer_result, 0, -1)
    parameters = np.copy(parameters, order='C')
    parameters = np.reshape(parameters, (-1, n_inputs), order='C')
    n_neurons = len(parameters)

    assert cnt == n_neurons

    cutoff = int(n_inputs * ratio)
    act = np.empty((n_neurons, cutoff), dtype=np.float32, order='C')
    idx_act = np.empty((n_neurons, cutoff), dtype=np.int32, order='C')
    bit_arr = np.empty((n_neurons, n_inputs * bits_per_input), dtype=np.bool, order='C')
    par_low_bound = np.empty((n_neurons, n_partitions), dtype=np.float32, order='C')

    _double_pointers = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
    index_lib.do_construct_parallel.restype = None
    index_lib.do_construct_parallel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                ctypes.c_int, ctypes.c_int,
                                                _double_pointers, _double_pointers,
                                                _double_pointers, _double_pointers,
                                                _double_pointers]
    parameters_pp = _get_double_pointers(parameters)
    act_pp = _get_double_pointers(act)
    idx_act_pp = _get_double_pointers(idx_act)
    bit_arr_pp = _get_double_pointers(bit_arr)
    par_low_bound_pp = _get_double_pointers(par_low_bound)

    index_lib.do_construct_parallel(bits_per_input, n_partitions, ratio, n_neurons, n_inputs,
                                    parameters_pp, act_pp, idx_act_pp, bit_arr_pp, par_low_bound_pp)

    bit_arr_packed = np.packbits(bit_arr, axis=1)

    return act, idx_act, bit_arr_packed, idx_idx, par_low_bound


def get_access_order(neuron_group, group_sample, n_inputs_in_partition_0, activations_with_idx_list, pointer_list):
    access_order_list = list()
    boundary_with_highest_activation_reached = [False] * len(neuron_group.neuron_idx_list)

    for neuron_id, activations_with_idx in enumerate(activations_with_idx_list):
        if pointer_list[neuron_id] is None:
            access_order_list.append(None)
            continue
        else:
            access_order_list.append(list())

        for round_cnt in range(n_inputs_in_partition_0):
            if pointer_list[neuron_id][0] - 1 >= 0:
                pointer_dec = pointer_list[neuron_id][0] - 1
            else:
                pointer_dec = pointer_list[neuron_id][0]

            if pointer_list[neuron_id][1] + 1 < n_inputs_in_partition_0:
                pointer_inc = pointer_list[neuron_id][1] + 1
            else:
                pointer_inc = pointer_list[neuron_id][1]

            if boundary_with_highest_activation_reached[neuron_id] \
                    or l2_dist(activations_with_idx[pointer_dec][0], group_sample[neuron_id]) \
                    <= l2_dist(activations_with_idx[pointer_inc][0], group_sample[neuron_id]):
                access_order_list[neuron_id].append(pointer_dec)
                if pointer_list[neuron_id][0] - 1 >= 0:
                    pointer_list[neuron_id][0] -= 1
            else:
                access_order_list[neuron_id].append(pointer_inc)
                if pointer_list[neuron_id][1] + 1 < n_inputs_in_partition_0:
                    pointer_list[neuron_id][1] += 1
                else:
                    boundary_with_highest_activation_reached[neuron_id] = True

    return access_order_list


def answer_query_with_guarantee(model, dataset, act, idx_act, bit_arr, idx_of_idx, par_low_bound,
                                input_sample_id, neuron_group, k, n_partitions, bits_per_input, BATCH_SIZE, batch_size,
                                where=None):
    layer_id = neuron_group.layer_id
    group_sample = get_group_sample(dataset, input_sample_id, layer_id, model, neuron_group)

    n_inputs = len(dataset)
    n_inputs_rerun = 1
    group_activation_cached = [None] * dataset.shape[0]
    group_activation_cached[input_sample_id] = group_sample
    heap = [(0.0, input_sample_id)]

    activations_with_idx_list, pointer_list = initialize_activations_and_pointers_for_phase_one(idx_of_idx,
                                                                                                input_sample_id,
                                                                                                group_sample,
                                                                                                neuron_group,
                                                                                                act,
                                                                                                idx_act)
    is_sample_in_partition_0 = [pointer is not None for pointer in pointer_list]
    n_inputs_in_partition_0 = len(activations_with_idx_list[0])

    access_order_list = get_access_order(neuron_group, group_sample, n_inputs_in_partition_0, activations_with_idx_list,
                                         pointer_list)

    print(f"input {input_sample_id}, size of neuron group {len(neuron_group.neuron_idx_list)}")

    exit_msg = None
    input_batch = set()
    ta_exited = False

    for round_cnt in range(n_inputs_in_partition_0):
        round_activations_with_idx = list()
        for neuron_id, activations_with_idx in enumerate(activations_with_idx_list):
            if access_order_list[neuron_id] is None:
                round_activations_with_idx.append(None)
            else:
                round_activations_with_idx.append(activations_with_idx[access_order_list[neuron_id][round_cnt]])

        for item in round_activations_with_idx:
            if item is None:
                continue
            activation, input_idx = item
            if group_activation_cached[input_idx] is None:
                if where is None:
                    pass
                else:
                    if not where(input_idx):
                        continue
                input_batch.add(input_idx)

        if len(input_batch) >= batch_size \
                or n_inputs_rerun + len(input_batch) == dataset.shape[0] \
                or round_cnt + 1 == n_inputs_in_partition_0:
            if len(input_batch) == 0:
                break
            run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, input_batch, k, layer_id,
                                     model, neuron_group, BATCH_SIZE)
            n_inputs_rerun += len(input_batch)
            input_batch = set()

        if len(input_batch) == 0 and len(heap) == k:
            round_activations = list()
            for round_activation_id, item in enumerate(round_activations_with_idx):
                if item is None:
                    round_activations.append(group_sample[round_activation_id])
                    continue
                activation, input_idx = item
                round_activations.append(activation)
            round_activations = np.array(round_activations).reshape(group_sample.shape)
            threshold = l2_dist(round_activations, group_sample)

            if heap[0] > (-threshold, n_inputs_in_partition_0):
                ta_exited = True
                break

    if ta_exited:
        return heap, exit_msg, is_sample_in_partition_0, n_inputs_rerun

    partitions_of_input = unpack_bits_and_get_input_partitions(idx_of_idx, neuron_group, bit_arr)

    input_batch, n_inputs_rerun = deal_with_remaining_inputs_in_partition_0(dataset, group_activation_cached,
                                                                            group_sample, heap, input_batch, k,
                                                                            layer_id, model, n_inputs_rerun,
                                                                            neuron_group, partitions_of_input,
                                                                            pointer_list, bits_per_input, BATCH_SIZE,
                                                                            where)

    bound_list, partition_pointer_list = initialize_bounds_and_pointers_for_phase_two(activations_with_idx_list,
                                                                                      input_sample_id, neuron_group,
                                                                                      partitions_of_input, pointer_list,
                                                                                      bits_per_input)

    lower_bound_of_partitions = get_lower_bound_of_partitions(idx_of_idx, neuron_group, par_low_bound)
    partition_access_order_list = get_partition_access_order_list(group_sample, n_partitions,
                                                                  neuron_group,
                                                                  lower_bound_of_partitions,
                                                                  partition_pointer_list)

    round_cnt = 0
    row_cnt = 0
    boundary_partition_processed = [[False, False] for idx in range(
        len(neuron_group.neuron_idx_list))]
    for neuron_id in range(len(neuron_group.neuron_idx_list)):
        if pointer_list[neuron_id] is not None:
            boundary_partition_processed[neuron_id][0] = True
    while n_inputs_rerun < dataset.shape[0]:

        inputs_for_neuron_list = list()
        for neuron_id, partition_of_input in enumerate(partitions_of_input):
            if round_cnt >= len(partition_access_order_list[neuron_id]):
                continue
            inputs_for_current_neuron = get_input_ids_by_partition_id(partition_of_input,
                                                                      partition_access_order_list[neuron_id][round_cnt],
                                                                      bits_per_input, n_inputs)
            inputs_for_neuron_list.append(inputs_for_current_neuron)
            add_inputs_to_batch(input_batch, inputs_for_current_neuron, group_activation_cached, where)

        row_cnt += (n_inputs - n_inputs_in_partition_0) // (n_partitions - 1)
        if len(input_batch) > 0:
            run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, input_batch, k, layer_id,
                                     model, neuron_group, BATCH_SIZE)
            n_inputs_rerun += len(input_batch)
            input_batch = set()

        for neuron_id in range(len(neuron_group.neuron_idx_list)):
            if partition_access_order_list[neuron_id][round_cnt] == 0 or (
                    n_inputs_in_partition_0 == 0 and partition_access_order_list[neuron_id][round_cnt] == 1):
                boundary_partition_processed[neuron_id][0] = True
            if partition_access_order_list[neuron_id][round_cnt] == n_partitions - 1:
                boundary_partition_processed[neuron_id][1] = True

        for idx in range(len(neuron_group.neuron_idx_list)):
            for input_id in inputs_for_neuron_list[idx]:
                if bound_list[idx] is None:
                    bound_list[idx] = [group_activation_cached[input_id][idx],
                                       group_activation_cached[input_id][idx]]
                else:
                    bound_list[idx][0] = min(bound_list[idx][0], group_activation_cached[input_id][idx])
                    bound_list[idx][1] = max(bound_list[idx][1], group_activation_cached[input_id][idx])

        if len(heap) == k:
            round_activations = np.array(group_sample)
            for idx in range(len(neuron_group.neuron_idx_list)):
                if boundary_partition_processed[idx][0] and not boundary_partition_processed[idx][1]:
                    round_activations[idx] = bound_list[idx][0]
                elif boundary_partition_processed[idx][1] and not boundary_partition_processed[idx][0]:
                    round_activations[idx] = bound_list[idx][1]
                elif pointer_list[idx] is None:
                    if l2_dist(bound_list[idx][0], group_sample[idx]) < l2_dist(bound_list[idx][1], group_sample[idx]):
                        round_activations[idx] = bound_list[idx][0]
                    else:
                        round_activations[idx] = bound_list[idx][1]
                else:
                    round_activations[idx] = bound_list[idx][0]

            threshold = l2_dist(round_activations, group_sample)

            if heap[0] > (-threshold, n_inputs_in_partition_0):
                ta_exited = True
                break

        round_cnt += 1

    if ta_exited:
        exit_msg = f"termination: phase 2, round {round_cnt}; inputs re-run: {n_inputs_rerun}"
    else:
        exit_msg = f"termination: none; inputs re-run: {n_inputs_rerun}"

    return heap, exit_msg, is_sample_in_partition_0, n_inputs_rerun


def get_group_sample(dataset, input_sample_id, layer_id, model, neuron_group):
    layer_result_sample = model.get_layer_result_by_layer_id([dataset[input_sample_id]], layer_id)[0]
    group_sample = get_group_activations_from_layer(neuron_group, layer_result_sample)
    return group_sample


def initialize_bounds_and_pointers_for_phase_two(activations_with_idx_list, input_sample_id, neuron_group,
                                                 partitions_of_input, pointer_list, bits_per_input):
    bound_list = list()
    partition_pointer_list = list()
    for neuron_id in range(len(neuron_group.neuron_idx_list)):
        if pointer_list[neuron_id] is None:
            partition_of_sample = get_partition_id_by_input_id(partitions_of_input[neuron_id], input_sample_id,
                                                               bits_per_input)
            partition_pointer_list.append([partition_of_sample, partition_of_sample])
            bound_list.append(None)
        else:
            partition_pointer_list.append(None)
            lower_bound = activations_with_idx_list[neuron_id][0][0]
            upper_bound = activations_with_idx_list[neuron_id][-1][0]
            bound_list.append([lower_bound, upper_bound])
    return bound_list, partition_pointer_list


def get_partition_access_order_list(group_sample, n_partitions, neuron_group, lower_bounds, partition_pointer_list):
    partition_access_order_list = list()
    for neuron_id in range(len(neuron_group.neuron_idx_list)):
        if partition_pointer_list[neuron_id] is None:
            partition_access_order_list.append([i for i in range(1, n_partitions)])
        else:
            partition_access_order_list.append([partition_pointer_list[neuron_id][0]])
            while True:
                pointer_dec = -1
                pointer_inc = -1
                if partition_pointer_list[neuron_id][0] - 1 >= 0:
                    pointer_dec = partition_pointer_list[neuron_id][0] - 1
                if partition_pointer_list[neuron_id][1] + 1 < n_partitions:
                    pointer_inc = partition_pointer_list[neuron_id][1] + 1

                if pointer_dec == -1 and pointer_inc == -1:
                    break
                else:
                    if pointer_dec == -1:
                        partition_access_order_list[neuron_id].append(pointer_inc)
                        partition_pointer_list[neuron_id][1] += 1
                    elif pointer_inc == -1:
                        partition_access_order_list[neuron_id].append(pointer_dec)
                        partition_pointer_list[neuron_id][0] -= 1
                    else:
                        if l2_dist(lower_bounds[neuron_id][pointer_dec], group_sample[neuron_id]) \
                                <= l2_dist(lower_bounds[neuron_id][pointer_inc], group_sample[neuron_id]):
                            partition_access_order_list[neuron_id].append(pointer_dec)
                            partition_pointer_list[neuron_id][0] -= 1
                        else:
                            partition_access_order_list[neuron_id].append(pointer_inc)
                            partition_pointer_list[neuron_id][1] += 1
    return partition_access_order_list


def deal_with_remaining_inputs_in_partition_0(dataset, group_activation_cached, group_sample, heap, input_batch, k,
                                              layer_id, model, n_inputs_rerun, neuron_group, partitions_of_input,
                                              pointer_list, bits_per_input, BATCH_SIZE, where):
    n_inputs = len(dataset)
    for idx, partition_of_input in enumerate(partitions_of_input):
        if pointer_list[idx] is not None:
            inputs_remaining = get_input_ids_by_partition_id(partition_of_input, 0, bits_per_input, n_inputs)
            add_inputs_to_batch(input_batch, inputs_remaining, group_activation_cached, where)
    if len(input_batch) > 0:
        run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, input_batch, k, layer_id,
                                 model, neuron_group, BATCH_SIZE)
        n_inputs_rerun += len(input_batch)
        print(f"partition 0: input batch into NN: {len(input_batch)}")
        input_batch = set()
    return input_batch, n_inputs_rerun


def get_lower_bound_of_partitions(idx_of_idx, neuron_group, par_low_bound):
    lower_bound_of_partitions = list()
    for neuron_idx in neuron_group.neuron_idx_list:
        lower_bound_of_partitions.append(par_low_bound[idx_of_idx[neuron_idx]])
    return lower_bound_of_partitions


def unpack_bits_and_get_input_partitions(idx_of_idx, neuron_group, bit_arr):
    partitions_of_input = list()
    for neuron_idx in neuron_group.neuron_idx_list:
        bits = np.unpackbits(bit_arr[idx_of_idx[neuron_idx]])
        partitions_of_input.append(bits)
    return partitions_of_input


def initialize_activations_and_pointers_for_phase_one(idx_of_idx, input_sample_id, group_sample,
                                                      neuron_group, act, idx_act):
    pointer_list = list()
    activations_with_idx_list = list()
    for i, neuron_idx in enumerate(neuron_group.neuron_idx_list):
        idx = idx_of_idx[neuron_idx]
        activations = act[idx]
        idx_activations = idx_act[idx]
        activations_with_idx = [(activations[i], idx_activations[i]) for i in range(len(activations))]
        activations_with_idx_list.append(activations_with_idx)
        sample_activation = group_sample[i]
        x = (sample_activation, input_sample_id)
        loc = binary_search(activations_with_idx, x)
        if loc == -1:
            pointer_list.append(None)
        else:
            pointer_list.append([loc + 1, loc])
    return activations_with_idx_list, pointer_list


def run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, input_batch, k, layer_id, model,
                             neuron_group, BATCH_SIZE):
    input_batch = list(input_batch)
    layer_result = get_layer_result_for_input_batch(model, dataset, input_batch, layer_id, BATCH_SIZE)
    for input_id, real_id in enumerate(input_batch):
        group_activation_cached[real_id] = get_group_activations_from_layer(neuron_group, layer_result[input_id])
    update_heap_from_cached_result(group_sample, heap, input_batch, k, group_activation_cached)


def add_inputs_to_batch(input_batch, inputs_to_add, cached_neuron_group_result, where):
    for input_id in inputs_to_add:
        if cached_neuron_group_result[input_id] is None:
            if where is None:
                pass
            else:
                if not where(input_id):
                    continue
            input_batch.add(input_id)


def update_heap_from_cached_result(group_sample, heap, input_batch, k, cached_neuron_group_result):
    for input_id, real_id in enumerate(input_batch):
        neuron_group_result = cached_neuron_group_result[real_id]
        dist = l2_dist(neuron_group_result, group_sample)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, real_id))
        elif (-dist, real_id) > heap[0]:
            heapq.heapreplace(heap, (-dist, real_id))


def load_all_indexes(filename_idx):
    start = timer()
    act = np.load(filename_idx[0])
    idx_act = np.load(filename_idx[1])
    bit_arr = np.load(filename_idx[2])
    par_low_bound = np.load(filename_idx[3])
    idx_idx = load_pickle(filename_idx[4])
    end = timer()
    query_time_load = end - start
    return par_low_bound, query_time_load, act, bit_arr, idx_act, idx_idx


def compute_persist_and_load_index(index_lib, n_inputs, dataset_name, layer_name, n_partitions, ratio, bits_per_input,
                                   layer_result):
    start = timer()
    act, idx_act, bit_arr, idx_idx, par_low_bound = construct_index(
        index_lib=index_lib,
        n_inputs=n_inputs,
        ratio=ratio,
        n_partitions=n_partitions,
        bits_per_input=bits_per_input,
        layer_result=layer_result)
    end = timer()
    prep_time_compute = end - start
    prep_time_dump, query_time_load, storage_size = persist_index(dataset_name, layer_name, n_partitions,
                                                                  ratio, par_low_bound, act,
                                                                  bit_arr, idx_act, idx_idx)

    return par_low_bound, act, bit_arr, idx_act, idx_idx, prep_time_compute, prep_time_dump, query_time_load, storage_size
