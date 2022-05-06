import gzip
import heapq
import os
import pickle
import shlex
import subprocess
from timeit import default_timer as timer

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import torch
from models.Mnist_torch import Mnist_torch
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras import utils

from models.Cifar10VGG import Cifar10VGG
from models.ImagenetResNet import ImagenetResNet50
from models.MnistVGG import MnistVGG


def get_neuron_result(layer_result, neuron, input_id=0):
    dim_1 = neuron // (layer_result.shape[2] * layer_result.shape[3])
    dim_2 = (neuron % (layer_result.shape[2] * layer_result.shape[3])) // layer_result.shape[3]
    dim_3 = (neuron % (layer_result.shape[2] * layer_result.shape[3])) % layer_result.shape[3]
    return layer_result[input_id][dim_1][dim_2][dim_3]


def get_topk_activation_by_layer_idx_in_batch(model, cur_input, k, layer, neuron, cur_input_id):
    layer_result = model.get_layer_result_by_layer_id(cur_input, layer)
    heap = get_topk_activation_heap_for_batch(cur_input_id, k, layer_result, neuron)
    return heap


def get_topk_activation_by_layer_name_in_batch(model, cur_input, k, layer_name, neuron, cur_input_id):
    layer_result = model.get_layer_result_by_layer_name(cur_input, layer_name)
    heap = get_topk_activation_heap_for_batch(cur_input_id, k, layer_result, neuron)
    return heap


def get_topk_activation_heap_for_batch(cur_input_id, k, layer_result, neuron):
    heap = []
    for input_id, real_id in enumerate(cur_input_id):
        neuron_result = get_neuron_result(layer_result, neuron, input_id)
        if len(heap) < k:
            heapq.heappush(heap, (neuron_result, real_id))
        elif (neuron_result, real_id) > heap[0]:
            heapq.heapreplace(heap, (neuron_result, real_id))
    return heap


def get_group_activations_from_layer(neuron_group, layer_activations):
    activations = list()
    for neuron_idx in neuron_group.neuron_idx_list:
        activations.append(layer_activations[neuron_idx])
    return np.asarray(activations)


def update_min_distance_heap(cur_input_id, model, dataset, dist_func, layer_sample, heap, k, neuron_group,
                             layer_result_dataset=None):
    if layer_result_dataset is None:
        cur_input = []
        for input_id in cur_input_id:
            cur_input.append(dataset[input_id])
        layer_result_batch = model.get_layer_result_by_layer_id(cur_input, neuron_group.layer_id)
    else:
        layer_result_batch = []
        for idx in cur_input_id:
            layer_result_batch.append(layer_result_dataset[idx])
        # layer_result_batch = np.array(layer_result_batch)

    for input_id, real_id in enumerate(cur_input_id):

        group_sample = get_group_activations_from_layer(neuron_group, layer_sample)
        group_input = get_group_activations_from_layer(neuron_group, layer_result_batch[input_id])

        dist = dist_func(group_sample, group_input)
        if len(heap) < k:
            heapq.heappush(heap, (-dist, real_id))
        elif (-dist, real_id) > heap[0]:
            heapq.heapreplace(heap, (-dist, real_id))


def update_max_norm_heap(cur_input_id, model, dataset, norm, heap, k, neuron_group, layer_result_dataset=None):
    if layer_result_dataset is None:
        cur_input = []
        for input_id in cur_input_id:
            cur_input.append(dataset[input_id])
        layer_result_batch = model.get_layer_result_by_layer_id(cur_input, neuron_group.layer_id)
    else:
        layer_result_batch = []
        for idx in cur_input_id:
            layer_result_batch.append(layer_result_dataset[idx])

    for input_id, real_id in enumerate(cur_input_id):

        group_input = get_group_activations_from_layer(neuron_group, layer_result_batch[input_id])

        dist = norm(group_input)
        if len(heap) < k:
            heapq.heappush(heap, (dist, real_id))
        elif (dist, real_id) > heap[0]:
            heapq.heapreplace(heap, (dist, real_id))


def initialize_data_model():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model = Cifar10VGG(train=False)
    print(model.model.summary())
    return model, x_test


def plot_cifar(X, y, idx):
    img = X[idx].reshape(32, 32, 3)
    plt.imshow(img, interpolation='nearest')
    plt.title('img: %d, true label: %d' % (idx, y[idx]))
    # plt.savefig("%d.png" % idx)
    plt.show()


def plot_mnist(X, y, idx, label_pred=None):
    img = X[idx].reshape(28, 28)
    plt.imshow(img, cmap='gray')
    if label_pred is None:
        label_pred = -1
    plt.title('img: %d, true label: %d, predicted: %d' % (idx, y[idx], label_pred))
    # plt.savefig("%d.png" % idx)
    plt.show()


def l2_dist(x, y):
    return np.sqrt(l2(x, y))


def l2_norm(x):
    return np.sqrt(np.sum(np.square(x)))


def l2(x, y):
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        if x.ndim != y.ndim:
            assert False
    return np.sum(np.square(x - y))


def cosine(x, y):
    x = x.flatten()
    y = y.flatten()
    return l2_dist(x / np.linalg.norm(x), y / np.linalg.norm(y))


def get_centroid(x):
    return np.mean(x, axis=0)


def gini(x):
    x = np.nan_to_num(x, False)
    x = x.flatten()
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 0.0000000001
    x = np.sort(x)
    index = np.arange(1, x.shape[0] + 1)
    n = x.shape[0]
    return (np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))


def l0_sparsity(x):
    x = np.nan_to_num(x, False)
    sparsity = 1.0 - (np.count_nonzero(x) / float(x.size))
    return sparsity


def load_imagenet_test_resnet_dataset_model():
    start = timer()
    x_test = np.load("/data/ilsvrc2012/ilsvrc2012_test.npy")
    end = timer()
    load_time = end - start
    model = ImagenetResNet50()
    return x_test, model, load_time


def load_imagenet_val_resnet_dataset_model():
    start = timer()
    x_val = np.load("/data/ilsvrc2012/ilsvrc2012_val_10000.npy")
    end = timer()
    load_time = end - start
    model = ImagenetResNet50()
    return x_val, model, load_time


def load_cifar10_vgg_dataset_model():
    start = timer()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    end = timer()
    load_time = end - start
    model = Cifar10VGG(train=False)
    return x_train, y_train, x_test, y_test, model, load_time


def load_mnist_vgg_dataset_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    model = MnistVGG(train=False)
    return x_train, y_train, x_test, y_test, model


def load_mnist_vgg_dataset_model_torch():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)
    y_train = utils.to_categorical(y_train)
    y_test = utils.to_categorical(y_test)

    model = Mnist_torch()
    return x_train, x_test, y_train, y_test, model


def equal_tuple(a, x, eps=1e-4):
    for i, j in zip(a, x):
        if abs(i - j) > eps:
            return False
    return True


def bisect_left(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def binary_search(a, x):
    pos = bisect_left(a, x)
    for i in range(max(0, pos - 3), min(len(a), pos + 2)):
        if equal_tuple(a[i], x):
            return i

    return -1


def is_power_of_two(n):
    if n == 0:
        return False
    while n != 1:
        if n % 2 != 0:
            return False
        n = n // 2

    return True


def gload(filename):
    clear_cache()
    file = gzip.GzipFile(filename, 'rb')
    res = pickle.load(file)
    file.close()
    return res


def gdump(obj, filename):
    file = gzip.GzipFile(filename, 'wb')
    pickle.dump(obj, file, -1)
    file.flush()
    os.fsync(file)
    file.close()


def load_pickle(filename):
    clear_cache()
    with open(filename, 'rb') as file:
        res = pickle.load(file)
    return res


def persist_pickle(filename, obj):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.flush()
        os.fsync(file)


def persist_numpy(filename, obj):
    with open(filename, 'wb') as file:
        np.save(file, obj)
        file.flush()
        os.fsync(file)


def get_bits(n, n_bits):
    return [n >> i & 1 for i in range(n_bits - 1, -1, -1)]


def get_layer_result_by_layer_name(model, x, layer_name, batch_size=None):
    if batch_size is None:
        res = model.get_layer_result_by_layer_name(x, layer_name)
    else:
        r = list()
        n = len(x)
        for i in range(n // batch_size + 1):
            if (i + 1) * batch_size >= n:
                layer_res = model.get_layer_result_by_layer_name(x[i * batch_size: n], layer_name)
            else:
                layer_res = model.get_layer_result_by_layer_name(x[i * batch_size: (i + 1) * batch_size], layer_name)

            r.append(layer_res)

            if (i + 1) * batch_size >= n:
                break

        res = np.concatenate(r, axis=0)

    return res


# batch_size not supported yet, should put all the data in one single array
def get_layer_result_by_layer_id(model, x, layer_id, batch_size=None):
    if batch_size is None:
        res = model.get_layer_result_by_layer_id(x, layer_id)
    else:
        r = list()
        n = len(x)
        for i in range(n // batch_size + 1):
            if (i + 1) * batch_size >= n:
                layer_res = model.get_layer_result_by_layer_id(x[i * batch_size: n], layer_id)
            else:
                layer_res = model.get_layer_result_by_layer_id(x[i * batch_size: (i + 1) * batch_size], layer_id)

            r.append(layer_res)

            if (i + 1) * batch_size >= n:
                break

        res = np.concatenate(r, axis=0)

    return res


# def get_layer_result_by_layer_id_torch(model, x, layer_id, batch_size=None):
#     if batch_size is None:
#         res = model.get_layer_result_by_layer_id(x, layer_id)
#     else:
#         r = list()
#         n = len(x)
#         for i in range(n // batch_size + 1):
#             if (i + 1) * batch_size >= n:
#                 layer_res = model.get_layer_result_by_layer_id(x[i * batch_size: n], layer_id)
#             else:
#                 layer_res = model.get_layer_result_by_layer_id(x[i * batch_size: (i + 1) * batch_size], layer_id)

#             r.append(layer_res)

#             if (i + 1) * batch_size >= n:
#                 break

#         res = np.concatenate(r, axis=0)

#     return res


def get_layer_results_by_layer_names(model, x, layer_names, batch_size=None):
    if batch_size is None:
        res = model.get_layer_results_by_layer_names(x, layer_names)
    else:
        r = list()
        for i in range(len(layer_names)):
            r.append(list())

        n = len(x)
        for i in range(n // batch_size + 1):
            if (i + 1) * batch_size >= n:
                layer_res = model.get_layer_results_by_layer_names(x[i * batch_size: n], layer_names)
            else:
                layer_res = model.get_layer_results_by_layer_names(x[i * batch_size: (i + 1) * batch_size], layer_names)

            for j in range(len(layer_res)):
                r[j].append(layer_res[j])

            if (i + 1) * batch_size >= n:
                break

        res = list()
        for i in range(len(r)):
            res.append(np.concatenate(r[i], axis=0))

    return res


def get_most_similar_input_based_on_neuron_group(model, dataset, k, neuron_group, dist_func, image_sample_id,
                                                 batch_size, layer_result_dataset=None):
    if batch_size is None:
        batch_size = 2000
    if layer_result_dataset is None:
        layer_sample = model.get_layer_result_by_layer_id([dataset[image_sample_id]], neuron_group.layer_id)[0]
    else:
        layer_sample = layer_result_dataset[image_sample_id]
    heap = []
    cur_input_id = []
    for i in range(dataset.shape[0]):
        cur_input_id.append(i)
        if (i + 1) % batch_size == 0 or i + 1 == dataset.shape[0]:
            update_min_distance_heap(cur_input_id, model, dataset, dist_func, layer_sample, heap, k, neuron_group,
                                     layer_result_dataset)
            cur_input_id = []

    return heap


def get_topk_images_producing_highest_activation_based_on_neuron_group(model, dataset, k, neuron_group, norm,
                                                                       batch_size, layer_result_dataset=None):
    if batch_size is None:
        batch_size = 2000
    heap = []
    cur_input_id = []
    for i in range(dataset.shape[0]):
        cur_input_id.append(i)
        if (i + 1) % batch_size == 0 or i + 1 == dataset.shape[0]:
            update_max_norm_heap(cur_input_id, model, dataset, norm, heap, k, neuron_group, layer_result_dataset)
            cur_input_id = []
    return heap


def get_topk_activations_given_images(model, dataset, image_ids, layer_name, k):
    res = list()
    image_samples = list()
    for image_sample_id in image_ids:
        image_samples.append(dataset[image_sample_id])
    layer_result_image_samples = get_layer_result_by_layer_name(model, image_samples, layer_name)
    for idx, image_sample_id in enumerate(image_ids):
        heap = list()
        for neuron_idx, activation in np.ndenumerate(layer_result_image_samples[idx]):
            if len(heap) < k:
                heapq.heappush(heap, (activation, neuron_idx))
            elif (activation, neuron_idx) > heap[0]:
                heapq.heapreplace(heap, (activation, neuron_idx))
        res.append(sorted(heap, reverse=True))
    return res


def get_rev_sorted_activations_given_images(model, dataset, image_ids, layer_name, nonzero, eps=5e-2):
    res = list()
    image_samples = list()
    for image_sample_id in image_ids:
        image_samples.append(dataset[image_sample_id])
    layer_result_image_samples = get_layer_result_by_layer_name(model, image_samples, layer_name)
    for idx, image_sample_id in enumerate(image_ids):
        act_neurons = list()
        for neuron_idx, activation in np.ndenumerate(layer_result_image_samples[idx]):
            if nonzero:
                if abs(activation) > eps:
                    act_neurons.append((activation, neuron_idx))
            else:
                act_neurons.append((activation, neuron_idx))
        res.append(sorted(act_neurons, reverse=True))
    return res


def warm_up_model(model, dataset):
    model.predict([dataset[0]])


def get_layer_result_for_image_batch(model, dataset, image_batch, layer_id, batch_size):
    cur_input = []
    for input_id in image_batch:
        cur_input.append(dataset[input_id])
    cur_input = torch.stack(cur_input)
    layer_result = get_layer_result_by_layer_id(model, cur_input, layer_id, batch_size)
    return layer_result


def get_partition_id_by_image_id(bit_array, image_id, bits_per_image):
    start_bit = image_id * bits_per_image
    end_bit = start_bit + bits_per_image
    res = 0
    for bit in bit_array[start_bit:end_bit]:
        res = (res << 1) | bit
    return res


def get_image_ids_by_partition_id(bit_array, partition_id, bits_per_image, n_images):
    images = set()
    partition_bits = get_bits(partition_id, bits_per_image)
    for image_id in range(n_images):
        start_bit = image_id * bits_per_image
        end_bit = start_bit + bits_per_image

        same = True
        for i, pos in enumerate(range(start_bit, end_bit)):
            if int(partition_bits[i]) != int(bit_array[pos]):
                same = False
                break
        if same:
            images.add(image_id)

    return images


def _get_double_pointers(x):
    return (x.__array_interface__['data'][0] + np.arange(x.shape[0]) * x.strides[0]).astype(np.uintp)


def warm_up_layer(model, dataset, layer_id, batch_size):
    for i in range(dataset.shape[0] // batch_size + 1):
        if (i + 1) * batch_size >= dataset.shape[0]:
            model.get_layer_result_by_layer_id(dataset[i * batch_size: dataset.shape[0]], layer_id)
        else:
            model.get_layer_result_by_layer_id(dataset[i * batch_size: (i + 1) * batch_size], layer_id)


def prod(tup):
    res = 1
    for ele in tup:
        res *= ele
    return res


def clear_cache():
    subprocess.run(shlex.split(
        "echo 1 > /proc/sys/vm/drop_caches"
    ), shell=True)


def prepare_layers_result_dataset(model, dataset, layer_names, all_layer_names, BATCH_SIZE):
    print("Preparing layers_result_dataset ...")
    layers_result_dataset = dict()
    for i in range(len(layer_names)):
        layer_id = all_layer_names.index(layer_names[i])
        if layer_id not in layers_result_dataset:
            layers_result_dataset[layer_id] = get_layer_result_by_layer_name(model, dataset, layer_names[i],
                                                                             batch_size=BATCH_SIZE)
    return layers_result_dataset


def persist_index(dataset_name, layer_name, n_partitions, ratio, par_low_bound, par_upp_bound, rev_act, rev_bit_arr,
                  rev_idx_act, rev_idx_idx):
    clear_cache()
    for i, obj in enumerate([rev_act, rev_idx_act, rev_bit_arr, par_low_bound, par_upp_bound, rev_idx_idx]):
        if i <= 4:
            filename = f"./index/{dataset_name}_{layer_name}_{n_partitions}_{ratio}_reverse_indices_{i}.npy"
            np.save(filename, obj)
        else:
            filename = f"./index/{dataset_name}_{layer_name}_{n_partitions}_{ratio}_reverse_indices_{i}.pickle"
            persist_pickle(filename, obj)


def evaluate(std, answer, eps=1e-4):
    std = sorted(std)
    answer = sorted(answer)
    std_image = [x[1] for x in std]
    answer_image = [x[1] for x in answer]

    tp = 0
    for i in range(len(answer)):
        if answer_image[i] in std_image or (i < len(std) and abs((answer[i][0] - std[i][0]) / std[i][0]) <= eps):
            tp += 1
    if len(answer) == 0:
        return 0.0, 0.0
    else:
        return tp / len(answer), tp / len(std)
