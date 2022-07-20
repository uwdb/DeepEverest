import ctypes
import heapq

import numpy as np

from utils import l2_dist, get_group_activations_from_layer, binary_search, \
    get_layer_result_for_image_batch, get_partition_id_by_image_id, get_image_ids_by_partition_id, \
    _get_double_pointers


def construct_index(index_lib, n_images, ratio, n_partitions, bits_per_image, layer_result):
    rev_idx_idx = dict()
    cnt = 0

    for neuron_idx, _ in np.ndenumerate(layer_result[0]):
        rev_idx_idx[neuron_idx] = cnt
        cnt += 1

    parameters = np.moveaxis(layer_result, 0, -1)
    parameters = np.copy(parameters, order='C')
    parameters = np.reshape(parameters, (-1, n_images), order='C')
    n_neurons = len(parameters)

    assert cnt == n_neurons

    cutoff = int(n_images * ratio)
    rev_act = np.empty((n_neurons, cutoff), dtype=np.float32, order='C')
    rev_idx_act = np.empty((n_neurons, cutoff), dtype=np.int32, order='C')
    rev_bit_arr = np.empty((n_neurons, n_images * bits_per_image), dtype=np.bool, order='C')
    par_low_bound = np.empty((n_neurons, n_partitions), dtype=np.float32, order='C')
    par_upp_bound = np.empty((n_neurons, n_partitions), dtype=np.float32, order='C')

    _double_pointers = np.ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags='C')
    index_lib.do_construct_parallel.restype = None
    index_lib.do_construct_parallel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                ctypes.c_int, ctypes.c_int,
                                                _double_pointers, _double_pointers,
                                                _double_pointers, _double_pointers,
                                                _double_pointers, _double_pointers]
    parameters_pp = _get_double_pointers(parameters)
    rev_act_pp = _get_double_pointers(rev_act)
    rev_idx_act_pp = _get_double_pointers(rev_idx_act)
    rev_bit_arr_pp = _get_double_pointers(rev_bit_arr)
    par_low_bound_pp = _get_double_pointers(par_low_bound)
    par_upp_bound_pp = _get_double_pointers(par_upp_bound)

    index_lib.do_construct_parallel(bits_per_image, n_partitions, ratio, n_neurons, n_images,
                                    parameters_pp, rev_act_pp, rev_idx_act_pp, rev_bit_arr_pp,
                                    par_low_bound_pp, par_upp_bound_pp)

    rev_bit_arr_packed = np.packbits(rev_bit_arr, axis=1)

    return rev_act, rev_idx_act, rev_bit_arr_packed, rev_idx_idx, par_low_bound, par_upp_bound


def get_access_order(neuron_group, group_sample, n_images_in_partition_0, activations_with_idx_list, pointer_list):
    access_order_list = list()
    boundary_with_highest_activation_reached = [False] * len(neuron_group.neuron_idx_list)

    for neuron_id, activations_with_idx in enumerate(activations_with_idx_list):
        if pointer_list[neuron_id] is None:
            access_order_list.append(None)
            continue
        else:
            access_order_list.append(list())

        for round_cnt in range(n_images_in_partition_0):
            if pointer_list[neuron_id][0] - 1 >= 0:
                pointer_dec = pointer_list[neuron_id][0] - 1
            else:
                pointer_dec = pointer_list[neuron_id][0]

            if pointer_list[neuron_id][1] + 1 < n_images_in_partition_0:
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
                if pointer_list[neuron_id][1] + 1 < n_images_in_partition_0:
                    pointer_list[neuron_id][1] += 1
                else:
                    boundary_with_highest_activation_reached[neuron_id] = True

    return access_order_list


def answer_query_with_guarantee(model, dataset, rev_act, rev_idx_act, rev_bit_arr, idx_of_rev_idx, par_low_bound,
                                par_upp_bound, image_sample_id, neuron_group, k, n_partitions, bits_per_image,
                                BATCH_SIZE, batch_size, where=None):
    layer_id = neuron_group.layer_id
    group_sample = get_group_sample(dataset, image_sample_id, layer_id, model, neuron_group)

    n_images = len(dataset)
    n_images_run = 1
    group_activation_cached = [None] * dataset.shape[0]
    group_activation_cached[image_sample_id] = group_sample
    heap = [(0.0, image_sample_id)]

    activations_with_idx_list, pointer_list = initialize_activations_and_pointers_for_phase_one(idx_of_rev_idx,
                                                                                                image_sample_id,
                                                                                                group_sample,
                                                                                                neuron_group,
                                                                                                rev_act,
                                                                                                rev_idx_act)
    is_sample_in_partition_0 = [pointer is not None for pointer in pointer_list]
    n_images_in_partition_0 = len(activations_with_idx_list[0])

    access_order_list = get_access_order(neuron_group, group_sample, n_images_in_partition_0, activations_with_idx_list,
                                         pointer_list)

    print(f"image {image_sample_id}, size of neuron group {len(neuron_group.neuron_idx_list)}")

    exit_msg = None
    image_batch = set()
    ta_exited = False

    for round_cnt in range(n_images_in_partition_0):
        round_activations_with_idx = list()
        for neuron_id, activations_with_idx in enumerate(activations_with_idx_list):
            if access_order_list[neuron_id] is None:
                round_activations_with_idx.append(None)
            else:
                round_activations_with_idx.append(activations_with_idx[access_order_list[neuron_id][round_cnt]])

        for item in round_activations_with_idx:
            if item is None:
                continue
            activation, image_idx = item
            if group_activation_cached[image_idx] is None:
                if where is None:
                    pass
                else:
                    if not where(image_idx):
                        continue
                image_batch.add(image_idx)

        if len(image_batch) >= batch_size \
                or n_images_run + len(image_batch) == dataset.shape[0] \
                or round_cnt + 1 == n_images_in_partition_0:
            if len(image_batch) == 0:
                break
            run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id,
                                     model, neuron_group, BATCH_SIZE)
            n_images_run += len(image_batch)
            image_batch = set()

        if len(image_batch) == 0 and len(heap) == k:
            round_activations = list()
            for round_activation_id, item in enumerate(round_activations_with_idx):
                if item is None:
                    round_activations.append(group_sample[round_activation_id])
                    continue
                activation, image_idx = item
                round_activations.append(activation)
            round_activations = np.array(round_activations).reshape(group_sample.shape)
            threshold = l2_dist(round_activations, group_sample)

            print(f"threshold: {threshold}, max in answer: {-heap[0][0]}, images run: {n_images_run}")
            if heap[0] > (-threshold, n_images_in_partition_0):
                print("======================= NTA exited =======================")
                exit_msg = f"termination: images run: {n_images_run}"
                ta_exited = True
                break

    if ta_exited:
        return heap, exit_msg, is_sample_in_partition_0, n_images_run

    partitions_of_image = unpack_bits_and_get_image_partitions(idx_of_rev_idx, neuron_group, rev_bit_arr)

    image_batch, n_images_run = deal_with_remaining_images_in_partition_0(dataset, group_activation_cached,
                                                                          group_sample, heap, image_batch, k,
                                                                          layer_id, model, n_images_run,
                                                                          neuron_group, partitions_of_image,
                                                                          pointer_list, bits_per_image, BATCH_SIZE,
                                                                          where)

    bound_list, partition_pointer_list = initialize_bounds_and_pointers_for_phase_two(activations_with_idx_list,
                                                                                      image_sample_id, neuron_group,
                                                                                      partitions_of_image, pointer_list,
                                                                                      bits_per_image)

    lower_bound_of_partitions = get_bound_of_partitions(idx_of_rev_idx, neuron_group, par_low_bound)
    upper_bound_of_partitions = get_bound_of_partitions(idx_of_rev_idx, neuron_group, par_upp_bound)
    partition_access_order_list = get_partition_access_order_list(group_sample, n_partitions, neuron_group,
                                                                  lower_bound_of_partitions, upper_bound_of_partitions,
                                                                  partition_pointer_list)

    round_cnt = 0
    row_cnt = 0
    boundary_partition_processed = [[False, False] for idx in range(
        len(neuron_group.neuron_idx_list))]
    for neuron_id in range(len(neuron_group.neuron_idx_list)):
        if pointer_list[neuron_id] is not None:
            boundary_partition_processed[neuron_id][0] = True
    while n_images_run < dataset.shape[0]:
        images_for_neuron_list = list()
        for neuron_id, partition_of_image in enumerate(partitions_of_image):
            if round_cnt >= len(partition_access_order_list[neuron_id]):
                continue
            images_for_current_neuron = get_image_ids_by_partition_id(partition_of_image,
                                                                      partition_access_order_list[neuron_id][round_cnt],
                                                                      bits_per_image, n_images)
            images_for_neuron_list.append(images_for_current_neuron)
            add_images_to_batch(image_batch, images_for_current_neuron, group_activation_cached, where)

        row_cnt += (n_images - n_images_in_partition_0) // (n_partitions - 1)
        if len(image_batch) > 0:
            run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id,
                                     model, neuron_group, BATCH_SIZE)
            n_images_run += len(image_batch)
            image_batch = set()

        for neuron_id in range(len(neuron_group.neuron_idx_list)):
            if partition_access_order_list[neuron_id][round_cnt] == 0 or (
                    n_images_in_partition_0 == 0 and partition_access_order_list[neuron_id][round_cnt] == 1):
                boundary_partition_processed[neuron_id][0] = True
            if partition_access_order_list[neuron_id][round_cnt] == n_partitions - 1:
                boundary_partition_processed[neuron_id][1] = True

        for idx in range(len(neuron_group.neuron_idx_list)):
            for image_id in images_for_neuron_list[idx]:
                if bound_list[idx] is None:
                    bound_list[idx] = [group_activation_cached[image_id][idx],
                                       group_activation_cached[image_id][idx]]
                else:
                    bound_list[idx][0] = min(bound_list[idx][0], group_activation_cached[image_id][idx])
                    bound_list[idx][1] = max(bound_list[idx][1], group_activation_cached[image_id][idx])

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
            print(
                f"threshold: {threshold}, max in answer: {-heap[0][0]}, images run: {n_images_run}")

            if heap[0] > (-threshold, n_images_in_partition_0):
                print("======================= NTA exited =======================")
                ta_exited = True
                break

        round_cnt += 1

    if ta_exited:
        exit_msg = f"termination: images run: {n_images_run}"
    else:
        exit_msg = f"termination: none; images run: {n_images_run}"

    return heap, exit_msg, is_sample_in_partition_0, n_images_run


def get_group_sample(dataset, image_sample_id, layer_id, model, neuron_group):
    layer_result_sample = model.get_layer_result_by_layer_id([dataset[image_sample_id]], layer_id)[0]
    group_sample = get_group_activations_from_layer(neuron_group, layer_result_sample)
    return group_sample


def initialize_bounds_and_pointers_for_phase_two(activations_with_idx_list, image_sample_id, neuron_group,
                                                 partitions_of_image, pointer_list, bits_per_image):
    bound_list = list()
    partition_pointer_list = list()
    for neuron_id in range(len(neuron_group.neuron_idx_list)):
        if pointer_list[neuron_id] is None:
            partition_of_sample = get_partition_id_by_image_id(partitions_of_image[neuron_id], image_sample_id,
                                                               bits_per_image)
            partition_pointer_list.append([partition_of_sample, partition_of_sample])
            bound_list.append(None)
        else:
            partition_pointer_list.append(None)
            lower_bound = activations_with_idx_list[neuron_id][0][0]
            upper_bound = activations_with_idx_list[neuron_id][-1][0]
            bound_list.append([lower_bound, upper_bound])
    return bound_list, partition_pointer_list


def get_partition_access_order_list(group_sample, n_partitions, neuron_group, lower_bounds, upper_bounds,
                                    partition_pointer_list):
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
                                <= l2_dist(upper_bounds[neuron_id][pointer_inc], group_sample[neuron_id]):
                            partition_access_order_list[neuron_id].append(pointer_dec)
                            partition_pointer_list[neuron_id][0] -= 1
                        else:
                            partition_access_order_list[neuron_id].append(pointer_inc)
                            partition_pointer_list[neuron_id][1] += 1
    return partition_access_order_list


def deal_with_remaining_images_in_partition_0(dataset, group_activation_cached, group_sample, heap, image_batch, k,
                                              layer_id, model, n_images_run, neuron_group, partitions_of_image,
                                              pointer_list, bits_per_image, BATCH_SIZE, where):
    n_images = len(dataset)
    for idx, partition_of_image in enumerate(partitions_of_image):
        if pointer_list[idx] is not None:
            images_remaining = get_image_ids_by_partition_id(partition_of_image, 0, bits_per_image, n_images)
            add_images_to_batch(image_batch, images_remaining, group_activation_cached, where)
    if len(image_batch) > 0:
        run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id,
                                 model, neuron_group, BATCH_SIZE)
        n_images_run += len(image_batch)
        print(f"partition 0: image batch into NN: {len(image_batch)}")
        image_batch = set()
    return image_batch, n_images_run


def get_bound_of_partitions(idx_of_rev_idx, neuron_group, bound):
    bound_of_partitions = list()
    for neuron_idx in neuron_group.neuron_idx_list:
        bound_of_partitions.append(bound[idx_of_rev_idx[neuron_idx]])
    return bound_of_partitions


def unpack_bits_and_get_image_partitions(idx_of_rev_idx, neuron_group, rev_bit_arr):
    partitions_of_image = list()
    for neuron_idx in neuron_group.neuron_idx_list:
        bits = np.unpackbits(rev_bit_arr[idx_of_rev_idx[neuron_idx]])
        partitions_of_image.append(bits)
    return partitions_of_image


def initialize_activations_and_pointers_for_phase_one(idx_of_rev_idx, image_sample_id, group_sample,
                                                      neuron_group, rev_act, rev_idx_act):
    pointer_list = list()
    activations_with_idx_list = list()
    for i, neuron_idx in enumerate(neuron_group.neuron_idx_list):
        idx = idx_of_rev_idx[neuron_idx]
        activations = rev_act[idx]
        idx_activations = rev_idx_act[idx]
        activations_with_idx = [(activations[i], idx_activations[i]) for i in range(len(activations))]
        activations_with_idx_list.append(activations_with_idx)
        sample_activation = group_sample[i]
        x = (sample_activation, image_sample_id)
        loc = binary_search(activations_with_idx, x)
        if loc == -1:
            pointer_list.append(None)
        else:
            pointer_list.append([loc + 1, loc])
    return activations_with_idx_list, pointer_list


def run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id, model,
                             neuron_group, BATCH_SIZE):
    image_batch = list(image_batch)
    layer_result = get_layer_result_for_image_batch(model, dataset, image_batch, layer_id, BATCH_SIZE)
    for input_id, real_id in enumerate(image_batch):
        group_activation_cached[real_id] = get_group_activations_from_layer(neuron_group, layer_result[input_id])
    update_heap_from_cached_result(group_sample, heap, image_batch, k, group_activation_cached)


def add_images_to_batch(image_batch, images_to_add, cached_neuron_group_result, where):
    for image_id in images_to_add:
        if cached_neuron_group_result[image_id] is None:
            if where is None:
                pass
            else:
                if not where(image_id):
                    continue
            image_batch.add(image_id)


def update_heap_from_cached_result(group_sample, heap, image_batch, k, cached_neuron_group_result):
    for input_id, real_id in enumerate(image_batch):
        neuron_group_result = cached_neuron_group_result[real_id]
        dist = l2_dist(neuron_group_result, group_sample)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, real_id))
        elif (-dist, real_id) > heap[0]:
            heapq.heapreplace(heap, (-dist, real_id))
