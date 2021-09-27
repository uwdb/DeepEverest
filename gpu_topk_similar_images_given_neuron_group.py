import os
from datetime import date
from random import random
from timeit import default_timer as timer

import statistics
import tensorflow as tf

from NeuronGroup import NeuronGroup
from utils import l2_dist, get_most_similar_input_based_on_neuron_group, warm_up_model, \
    get_layer_result_by_layer_name, load_imagenet_val_resnet_dataset_model, get_topk_activations_given_images


def recompute_all():
    print("images:", image_sample_ids)
    print("images:", image_sample_ids, file=fout)

    for neuron_group_id, layer_name in enumerate(layer_names):
        neuron_group_list = neuron_groups[neuron_group_id]
        query_times = list()
        for i, image_sample_id in enumerate(image_sample_ids):
            print(f"running {image_sample_id} ...")
            start = timer()
            top = get_most_similar_input_based_on_neuron_group(model, dataset, k_global, neuron_group_list[i],
                                                               l2_dist, image_sample_id, batch_size=BATCH_SIZE)
            end = timer()
            query_time = end - start
            query_times.append(query_time)

        print(f"recompute-all {layer_name}, size of neuron group {len(neuron_group_list[0].neuron_idx_list)}")
        print(f"query time median: {statistics.median(query_times)}, min: {min(query_times)}, max: {max(query_times)}")
        print(f"query times: {query_times}")
        print("")

        print(f"recompute-all {layer_name}, size of neuron group {len(neuron_group_list[0].neuron_idx_list)}",
              file=fout)
        print(f"query time median: {statistics.median(query_times)}, min: {min(query_times)}, max: {max(query_times)}",
              file=fout)
        print(f"query times: {query_times}",
              file=fout)
        print("", file=fout)


def store_all():
    print("images:", image_sample_ids)
    print("images:", image_sample_ids, file=fout)

    for neuron_group_id, layer_name in enumerate(layer_names):
        layer_result_dataset = get_layer_result_by_layer_name(model, dataset, layer_name, batch_size=BATCH_SIZE)
        neuron_group_list = neuron_groups[neuron_group_id]

        query_times = list()
        for i, image_sample_id in enumerate(image_sample_ids):
            start = timer()
            top = get_most_similar_input_based_on_neuron_group(model, dataset, k_global, neuron_group_list[i], l2_dist,
                                                               image_sample_id, BATCH_SIZE, layer_result_dataset)
            end = timer()
            query_time = end - start
            query_times.append(query_time)

        print(f"store-all {layer_name}, size of neuron group {len(neuron_group_list[0].neuron_idx_list)}")
        print(f"query time median: {statistics.median(query_times)}, min: {min(query_times)}, max: {max(query_times)}")
        print(f"query times: {query_times}")
        print("")

        print(f"store-all {layer_name}, size of neuron group {len(neuron_group_list[0].neuron_idx_list)}",
              file=fout)
        print(f"query time median: {statistics.median(query_times)}, min: {min(query_times)}, max: {max(query_times)}",
              file=fout)
        print(f"query times: {query_times}",
              file=fout)
        print("", file=fout)


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
    imagenet = "imagenet"
    group = "top"
    dataset, model, dataset_loading_time = load_imagenet_val_resnet_dataset_model()
    BATCH_SIZE = 64
    all_layer_names = [layer.name for layer in model.model.layers]
    batch_size, image_sample_ids, k_global, layer_names, n_partitions_list, neuron_groups, ratios = \
        imagenet_prepare_experiments_top_activation_neuron_group(model, dataset, all_layer_names)

    today = date.today()
    date_str = today.strftime("%m%d")

    output_filename = f"{date_str}_{imagenet}_naive_{group}_group_{BATCH_SIZE}_benchmark.txt"
    fout = open(output_filename, "w")

    warm_up_model(model, dataset)
    print(model.model.summary())
    all_layer_names = [layer.name for layer in model.model.layers]
    print("dataset loading time:", dataset_loading_time)
    print("BATCH_SIZE:", BATCH_SIZE)
    print(imagenet, group)

    print(f"dataset loading time: {dataset_loading_time}", file=fout)
    print(f"BATCH_SIZE: {BATCH_SIZE}", file=fout)
    print(imagenet, group, file=fout)

    recompute_all()
    store_all()

    fout.close()
