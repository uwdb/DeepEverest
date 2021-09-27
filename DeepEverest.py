import ctypes
import gc
import heapq
import math
import os
import random
import statistics
from datetime import date
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf

from NeuronGroup import NeuronGroup
from utils import l2_dist, get_group_activations_from_layer, binary_search, \
    get_most_similar_input_based_on_neuron_group, warm_up_model, \
    get_layer_result_for_image_batch, get_partition_id_by_image_id, get_image_ids_by_partition_id, \
    _get_double_pointers, load_imagenet_val_resnet_dataset_model, get_layer_result_by_layer_id, load_pickle, \
    get_topk_activations_given_images, persist_index, evaluate, clear_cache


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
    n_images_rerun = 1
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
    print("entering phase 1 ...")

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
                or n_images_rerun + len(image_batch) == dataset.shape[0] \
                or round_cnt + 1 == n_images_in_partition_0:
            if len(image_batch) == 0:
                break
            run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id,
                                     model, neuron_group, BATCH_SIZE)
            n_images_rerun += len(image_batch)
            print(f"phase 1, round {round_cnt}: images in batch fed to NN: {len(image_batch)}")
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

            print(
                f"phase 1, round {round_cnt}, threshold: {threshold}, max in answer: {-heap[0][0]}, images re-run: {n_images_rerun}")
            if heap[0] > (-threshold, n_images_in_partition_0):
                print("======================= TA exited =======================")
                exit_msg = f"termination: phase 1, round {round_cnt}; images re-run: {n_images_rerun}"
                ta_exited = True
                break

    if ta_exited:
        return heap, exit_msg, is_sample_in_partition_0, n_images_rerun

    partitions_of_image = unpack_bits_and_get_image_partitions(idx_of_rev_idx, neuron_group, rev_bit_arr)

    image_batch, n_images_rerun = deal_with_remaining_images_in_partition_0(dataset, group_activation_cached,
                                                                            group_sample, heap, image_batch, k,
                                                                            layer_id, model, n_images_rerun,
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
    while n_images_rerun < dataset.shape[0]:
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
            n_images_rerun += len(image_batch)
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
                f"phase 2, round {round_cnt}, threshold: {threshold}, max in answer: {-heap[0][0]}, images re-run: {n_images_rerun}, row count: {row_cnt}")

            if heap[0] > (-threshold, n_images_in_partition_0):
                print("======================= TA exited =======================")
                ta_exited = True
                break

        round_cnt += 1

    if ta_exited:
        exit_msg = f"termination: phase 2, round {round_cnt}; images re-run: {n_images_rerun}"
    else:
        exit_msg = f"termination: none; images re-run: {n_images_rerun}"

    return heap, exit_msg, is_sample_in_partition_0, n_images_rerun


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
                                              layer_id, model, n_images_rerun, neuron_group, partitions_of_image,
                                              pointer_list, bits_per_image, BATCH_SIZE, where):
    n_images = len(dataset)
    for idx, partition_of_image in enumerate(partitions_of_image):
        if pointer_list[idx] is not None:
            images_remaining = get_image_ids_by_partition_id(partition_of_image, 0, bits_per_image, n_images)
            add_images_to_batch(image_batch, images_remaining, group_activation_cached, where)
    if len(image_batch) > 0:
        run_nn_and_update_things(dataset, group_activation_cached, group_sample, heap, image_batch, k, layer_id,
                                 model, neuron_group, BATCH_SIZE)
        n_images_rerun += len(image_batch)
        print(f"partition 0: image batch into NN: {len(image_batch)}")
        image_batch = set()
    return image_batch, n_images_rerun


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


def benchmark(dataset):
    global all_layer_names, bits_per_image, BATCH_SIZE

    fwrite_benchmark_data(fout, batch_size, image_sample_ids, n_partitions_list, ratios)
    fwrite_benchmark_data(fout_verbose, batch_size, image_sample_ids, n_partitions_list, ratios)

    last_layer_id = -1
    layer_result = None

    for neuron_group_id, layer_name in enumerate(layer_names):
        layer_id = all_layer_names.index(layer_name)
        neuron_group_list = neuron_groups[neuron_group_id]
        if layer_id != last_layer_id:
            print(f"preparing layer_result for {layer_name} ...")
            layer_result = None
            gc.collect()
            layer_result = get_layer_result_by_layer_id(model, dataset, layer_id, batch_size=batch_size)
        else:
            print(f"using layer_result from the previous set of experiments for {layer_name} ...")

        answer_std = list()

        print(
            f"getting standard answers for {layer_name}, size of neuron group {len(neuron_group_list[0].neuron_idx_list)}")

        time_brute_force_list = list()
        for i, image_sample_id in enumerate(image_sample_ids):
            start = timer()
            top_k = get_most_similar_input_based_on_neuron_group(model, dataset, k_global, neuron_group_list[i],
                                                                 l2_dist, image_sample_id, batch_size, layer_result)
            end = timer()
            answer_std.append(sorted(top_k))
            time_brute_force = end - start
            time_brute_force_list.append(time_brute_force)
        print("store-everything (excl. load) query time median:", statistics.median(time_brute_force_list))

        precision_list_list = list()
        recall_list_list = list()
        query_time_median_list_list = list()
        query_time_min_list_list = list()
        query_time_max_list_list = list()
        query_times_list_list = list()
        query_time_load_list_list = list()
        exit_msgs_list_list = list()
        prep_time_compute_list_list = list()
        prep_time_dump_list_list = list()
        storage_list_list = list()
        is_in_partition_0_list_list_list = list()
        n_images_rerun_median_list_list = list()

        for n_partitions in n_partitions_list:

            precision_list = list()
            recall_list = list()
            query_time_median_list = list()
            query_time_min_list = list()
            query_time_max_list = list()
            query_times_list = list()
            query_time_load_list = list()
            exit_msgs_list = list()
            prep_time_compute_list = list()
            prep_time_dump_list = list()
            storage_list = list()
            is_in_partition_0_list_list = list()
            n_images_rerun_median_list = list()

            bits_per_image = math.ceil(math.log(n_partitions, 2))

            for ratio in ratios:
                print(f"n_partitions={n_partitions}, ratio={ratio}, start pre-processing ...")

                par_low_bound, par_upp_bound, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx = preprocess(layer_name,
                                                                                                          n_partitions,
                                                                                                          ratio,
                                                                                                          bits_per_image,
                                                                                                          layer_result,
                                                                                                          prep_time_compute_list,
                                                                                                          prep_time_dump_list,
                                                                                                          query_time_load_list,
                                                                                                          storage_list)
                # execution
                answer = list()
                exit_msgs = list()
                query_times = list()
                is_in_partition_0_list = list()
                n_images_rerun_list = list()
                for i, image_sample_id in enumerate(image_sample_ids):
                    start = timer()
                    top_k, exit_msg, is_in_partition_0, n_images_rerun = \
                        answer_query_with_guarantee(model, dataset, rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx,
                                                    par_low_bound, par_upp_bound, image_sample_id,
                                                    neuron_group_list[i], k_global, n_partitions, bits_per_image,
                                                    BATCH_SIZE=BATCH_SIZE, batch_size=batch_size)
                    end = timer()
                    query_time = end - start
                    top_k = sorted(top_k)

                    is_in_partition_0_list.append(is_in_partition_0)
                    query_times.append(query_time)
                    answer.append(top_k)
                    exit_msgs.append(exit_msg)
                    n_images_rerun_list.append(n_images_rerun)
                    print(f"==== image {image_sample_id}, query time: {query_time}, msg: {exit_msg} ====")

                # evaluation
                evaluate_then_collect(answer, answer_std, exit_msgs, exit_msgs_list, image_sample_ids,
                                      is_in_partition_0_list, is_in_partition_0_list_list, precision_list,
                                      query_time_max_list, query_time_median_list, query_time_min_list, query_times,
                                      query_times_list, recall_list, n_images_rerun_median_list,
                                      n_images_rerun_list)

                print(
                    f"{layer_name}, n_partitions={n_partitions}, ratio={ratio}, precision: {precision_list[-1]}, query time median: {statistics.median(query_times)}")

            precision_list_list.append(precision_list)
            recall_list_list.append(recall_list)
            query_time_median_list_list.append(query_time_median_list)
            query_time_min_list_list.append(query_time_min_list)
            query_time_max_list_list.append(query_time_max_list)
            query_times_list_list.append(query_times_list)
            query_time_load_list_list.append(query_time_load_list)
            exit_msgs_list_list.append(exit_msgs_list)
            prep_time_compute_list_list.append(prep_time_compute_list)
            prep_time_dump_list_list.append(prep_time_dump_list)
            storage_list_list.append(storage_list)
            is_in_partition_0_list_list_list.append(is_in_partition_0_list_list)
            n_images_rerun_median_list_list.append(n_images_rerun_median_list)

            print_verbose(n_partitions,
                          exit_msgs_list, image_sample_ids, is_in_partition_0_list_list, layer_name, neuron_group_list,
                          precision_list, prep_time_compute_list, prep_time_dump_list, query_time_load_list,
                          query_time_max_list, query_time_median_list, query_time_min_list, query_times_list,
                          recall_list, storage_list, n_images_rerun_median_list)

        fwrite_non_verbose(fout, exit_msgs_list_list, image_sample_ids, is_in_partition_0_list_list_list, layer_name,
                           neuron_group_list, precision_list_list, prep_time_compute_list_list,
                           prep_time_dump_list_list, query_time_load_list_list, query_time_max_list_list,
                           query_time_median_list_list, query_time_min_list_list, query_times_list_list,
                           recall_list_list, storage_list_list, n_images_rerun_median_list_list)

        fwrite_verbose(fout_verbose, exit_msgs_list_list, image_sample_ids, is_in_partition_0_list_list_list,
                       layer_name, neuron_group_list, precision_list_list, prep_time_compute_list_list,
                       prep_time_dump_list_list, query_time_load_list_list, query_time_max_list_list,
                       query_time_median_list_list, query_time_min_list_list, query_times_list_list,
                       recall_list_list, storage_list_list, n_images_rerun_median_list_list)

        last_layer_id = layer_id


def evaluate_then_collect(answer, answer_std, exit_msgs, exit_msgs_list, image_sample_ids, is_in_partition_0_list,
                          is_in_partition_0_list_list, precision_list, query_time_max_list, query_time_median_list,
                          query_time_min_list, query_times, query_times_list, recall_list,
                          n_images_rerun_median_list, n_images_rerun_list):
    sum_precision = 0.0
    sum_recall = 0.0
    for i in range(len(image_sample_ids)):
        precision, recall = evaluate(answer_std[i], answer[i])
        sum_precision += precision
        sum_recall += recall
        print(image_sample_ids[i], precision, recall)
        if precision < 1.0:
            print(str(answer_std[i]))
            print(str(answer[i]))
    precision_list.append(sum_precision / len(image_sample_ids))
    recall_list.append(sum_recall / len(image_sample_ids))
    query_time_median_list.append(statistics.median(query_times))
    query_time_min_list.append(min(query_times))
    query_time_max_list.append(max(query_times))
    query_times_list.append(query_times)
    exit_msgs_list.append(exit_msgs)
    is_in_partition_0_list_list.append(is_in_partition_0_list)
    n_images_rerun_median_list.append(statistics.median(n_images_rerun_list))


def preprocess(layer_name, n_partitions, ratio, bits_per_image, layer_result, prep_time_compute_list,
               prep_time_dump_list, query_time_load_list, storage_list):
    if LOAD_INDEX:
        try:
            storage_size = 0.0
            filename_rev_idx = list()
            for i in range(6):
                if i <= 4:
                    filename = f"/data/{imagenet}_{layer_name}_{n_partitions}_{ratio}_indices_{i}.npy"
                else:
                    filename = f"/data/{imagenet}_{layer_name}_{n_partitions}_{ratio}_indices_{i}.pickle"
                storage_size += os.stat(filename).st_size / 1024 / 1024
                filename_rev_idx.append(filename)
            par_low_bound, par_upp_bound, query_time_load, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx = load_all_indices(
                filename_rev_idx)
            prep_time_compute = None
            prep_time_dump = None
        except FileNotFoundError:
            par_low_bound, par_upp_bound, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx, \
            prep_time_compute, prep_time_dump, query_time_load, storage_size = compute_persist_and_load_index(
                layer_name, n_partitions, ratio, bits_per_image, layer_result)
    else:
        par_low_bound, par_upp_bound, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx, \
        prep_time_compute, prep_time_dump, query_time_load, storage_size = compute_persist_and_load_index(
            layer_name, n_partitions, ratio, bits_per_image, layer_result)
    preprocessing_collect(prep_time_compute, prep_time_compute_list,
                          prep_time_dump, prep_time_dump_list,
                          query_time_load, query_time_load_list,
                          storage_size, storage_list)
    return par_low_bound, par_upp_bound, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx


def load_all_indices(filename_rev_idx):
    clear_cache()
    start = timer()
    rev_act = np.load(filename_rev_idx[0])
    rev_idx_act = np.load(filename_rev_idx[1])
    rev_bit_arr = np.load(filename_rev_idx[2])
    par_low_bound = np.load(filename_rev_idx[3])
    par_upp_bound = np.load(filename_rev_idx[4])
    rev_idx_idx = load_pickle(filename_rev_idx[5])
    end = timer()
    query_time_load = end - start
    return par_low_bound, par_upp_bound, query_time_load, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx


def preprocessing_collect(prep_time_compute, prep_time_compute_list, prep_time_dump, prep_time_dump_list,
                          query_time_load, query_time_load_list, storage_size, storage_list):
    prep_time_compute_list.append(prep_time_compute)
    print("pre-processing compute time:", prep_time_compute)
    prep_time_dump_list.append(prep_time_dump)
    print("pre-processing dump time:", prep_time_dump)
    storage_list.append(storage_size)
    print("storage (MB):", storage_size)
    query_time_load_list.append(query_time_load)
    print("query_time load:", query_time_load)


def compute_persist_and_load_index(layer_name, n_partitions, ratio, bits_per_image, layer_result):
    start = timer()
    rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx, par_low_bound, par_upp_bound = construct_index(
        index_lib=index_lib,
        n_images=n_images,
        ratio=ratio,
        n_partitions=n_partitions,
        bits_per_image=bits_per_image,
        layer_result=layer_result)
    end = timer()
    prep_time_compute = end - start
    print("pre-processing compute done ...")
    prep_time_dump, query_time_load, storage_size = persist_index(imagenet, layer_name, n_partitions, ratio,
                                                                  par_low_bound, par_upp_bound, rev_act,
                                                                  rev_bit_arr, rev_idx_act, rev_idx_idx)

    return par_low_bound, par_upp_bound, rev_act, rev_bit_arr, rev_idx_act, rev_idx_idx, prep_time_compute, \
           prep_time_dump, query_time_load, storage_size


def fwrite_benchmark_data(f, batch_size, image_sample_ids, n_partitions_list, ratios):
    print(f"n_partitions: {n_partitions_list}", file=f)
    print(f"ratios: {ratios}", file=f)
    print(f"TA batch_size: {batch_size}", file=f)
    print(f"random images: {image_sample_ids}", file=f)
    print("", file=f)
    print("", file=f)


def print_verbose(n_partitions, exit_msgs_list, image_sample_ids, is_in_partition_0_list_list, layer_name,
                  neuron_group_list, precision_list, prep_time_compute_list, prep_time_dump_list, query_time_load_list,
                  query_time_max_list, query_time_median_list, query_time_min_list, query_times_list, recall_list,
                  storage_list, n_images_rerun_median_list):
    print("")
    print(f"Result for {layer_name}, n_partitions {n_partitions}")
    for i, neuron_group in enumerate(neuron_group_list):
        print(image_sample_ids[i], neuron_group.neuron_idx_list)
    print("----------------------------------------------")
    print(precision_list)
    print(query_time_median_list)
    print(n_images_rerun_median_list)
    print("----------------------------------------------")
    print("")
    print("")


def fwrite_non_verbose(f, exit_msgs_list_list, image_sample_ids, is_in_partition_0_list_list_list, layer_name,
                       neuron_group_list, precision_list_list, prep_time_compute_list_list, prep_time_dump_list_list,
                       query_time_load_list_list, query_time_max_list_list, query_time_median_list_list,
                       query_time_min_list_list, query_times_list_list, recall_list_list, storage_list_list,
                       n_images_rerun_median_list_list):
    print(f"Result for {layer_name}, neuron group size {len(neuron_group_list[0].neuron_idx_list)}", file=f)
    print(f"precision: {precision_list_list}", file=f)
    print(f"query_time_median excl. load (s): {query_time_median_list_list}", file=f)
    print(f"n_images_rerun_median: {n_images_rerun_median_list_list}", file=f)
    print("", file=f)
    print(f"query_times (s): {query_times_list_list}", file=f)
    print(f"query_time load (s): {query_time_load_list_list}", file=f)
    print(f"storage for one layer (MB): {storage_list_list}", file=f)
    print("", file=f)
    print("", file=f)


def fwrite_verbose(f, exit_msgs_list_list, image_sample_ids, is_in_partition_0_list_list_list, layer_name,
                   neuron_group_list, precision_list_list, prep_time_compute_list_list, prep_time_dump_list_list,
                   query_time_load_list_list, query_time_max_list_list, query_time_median_list_list,
                   query_time_min_list_list, query_times_list_list, recall_list_list, storage_list_list,
                   n_images_rerun_median_list_list):
    print(f"Result for {layer_name}", file=f)
    for i, neuron_group in enumerate(neuron_group_list):
        print(f"image_id: {image_sample_ids[i]}, neuron group: {neuron_group.neuron_idx_list}", file=f)
    print("", file=f)
    print(f"precision: {precision_list_list}", file=f)
    print(f"recall: {recall_list_list}", file=f)
    print(f"query_time_median excl. load (s): {query_time_median_list_list}", file=f)
    print(f"n_images_rerun_median: {n_images_rerun_median_list_list}", file=f)
    print(f"query_time_min excl. load (s): {query_time_min_list_list}", file=f)
    print(f"query_time_max excl. load (s): {query_time_max_list_list}", file=f)
    print(f"query_time load (s): {query_time_load_list_list}", file=f)
    print(f"prep_time_compute (for one layer): {prep_time_compute_list_list}", file=f)
    print(f"prep_time_dump (for one layer): {prep_time_dump_list_list}", file=f)
    print(f"storage for one layer (MB): {storage_list_list}", file=f)
    print(f"exit_msg: {exit_msgs_list_list}", file=f)
    print(f"is_image_sample_in_partition_0: {is_in_partition_0_list_list_list}", file=f)
    print(f"query_times (s): {query_times_list_list}", file=f)
    print("", file=f)
    print("", file=f)


def imagenet_prepare_experiments_top_activation_neuron_group(model, dataset, all_layer_names):
    image_sample_ids = random.choices(range(len(dataset)), k=5)
    layer_names = ["activation_2", "activation_2", "activation_2",
                   "activation_25", "activation_25", "activation_25",
                   "activation_48", "activation_48", "activation_48"]
    neuron_group_sizes = [1, 3, 10, 1, 3, 10, 1, 3, 10]
    neuron_groups = get_top_neuron_group_for_images(model, dataset, layer_names, neuron_group_sizes, image_sample_ids,
                                                    all_layer_names)
    k_global = 20
    ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
    n_partitions_list = [2, 4, 8, 16, 32, 64, 128, 256]
    batch_size = 64
    return batch_size, image_sample_ids, k_global, layer_names, n_partitions_list, neuron_groups, ratios


def get_top_neuron_group_for_images(model, dataset, layer_names, neuron_group_sizes, image_sample_ids, all_layer_names):
    neuron_groups = list()
    for i, layer_name in enumerate(layer_names):
        top_activations = get_topk_activations_given_images(model, dataset, image_sample_ids, layer_name,
                                                            neuron_group_sizes[i])
        neuron_group_list = list()
        for j in range(len(image_sample_ids)):
            neuron_group_list.append(NeuronGroup(model=model.model, layer_id=all_layer_names.index(layer_name),
                                                 neuron_idx_list=[x[1] for x in top_activations[j]]))
        neuron_groups.append(neuron_group_list)
    return neuron_groups


if __name__ == '__main__':
    assert tf.test.is_gpu_available()
    assert tf.test.is_built_with_cuda()
    os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
    random.seed(1001)
    lib_file = "./index/build/lib.linux-x86_64-3.6/deepeverst_index.cpython-36m-x86_64-linux-gnu.so"
    index_lib = ctypes.CDLL(lib_file)

    imagenet = "imagenet"
    group = "top"
    dataset, model, dataset_loading_time = load_imagenet_val_resnet_dataset_model()
    BATCH_SIZE = 64
    all_layer_names = [layer.name for layer in model.model.layers]
    batch_size, image_sample_ids, k_global, layer_names, n_partitions_list, neuron_groups, ratios = \
        imagenet_prepare_experiments_top_activation_neuron_group(model, dataset, all_layer_names)

    n_images = len(dataset)
    LOAD_INDEX = False

    warm_up_model(model, dataset)
    print(model.model.summary())
    print("image dataset loading time:", dataset_loading_time)

    today = date.today()
    date_str = today.strftime("%m%d")

    output = f"{date_str}_{imagenet}_{group}_group_{batch_size}_benchmark.txt"
    output_verbose = f"{date_str}_{imagenet}_{group}_group_{batch_size}_benchmark_verbose.txt"
    fout = open(output, "w")
    fout_verbose = open(output_verbose, "w")

    print(imagenet, group, file=fout_verbose)
    print("image dataset loading time:", dataset_loading_time, file=fout_verbose)

    benchmark(dataset)

    fout.close()
