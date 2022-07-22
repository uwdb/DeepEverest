from operator import is_
import numpy as np
import torch
from models.BaseModel import BaseModel
import ctypes
import helper
import math
import heapq
import pickle

# the Deep Everest API class
# use examples are shown in the exmaple_api notebook
# for PyTorch user, user can create model using this class, and pass in the parameters
# for TensorFlow user, user can create a super class as shown in the MnistVGG_api script

# this class extends the BaseModel which ultimately represents the model
# model: the trained model of TensorFlow or PyTorch
# is_torch: True if model is pytorch, False if model is TensorFlow
# lib_file: the file path of the compiled c library
# dataset: the whole input dataset that user would like to search on
#          torch tensor for PyTorch, numpy array for TensorFlow
# n_par: number of partitions, default is 64
# bs: batch size, PyTorch should use None
# ratio: ratio, default is 0.05
class DeepEverest(BaseModel):

    def __init__(self, model, is_torch, lib_file, dataset, n_partitions=64, batch_size=None, ratio=0.05):
        BaseModel.__init__(self, model, is_torch)
        helper.is_torch = is_torch
        self.model = model
        self.lib_file = lib_file
        self.index_lib = ctypes.CDLL(lib_file)
        self.dataset = dataset
        self.n_images = len(dataset)
        self.n_partitions = n_partitions
        self.batch_size = batch_size
        if batch_size == None:
            self.batch_size_answer = 64
        else:
            self.batch_size_answer = batch_size
        self.ratio = ratio
        self.bits_per_image = math.ceil(math.log(n_partitions, 2))
        self.index_map = {}
        self.current_index_layer_id = None
    
    
    # return if the model is pytorch or not
    def is_torch(self):
        return self.is_torch

    
    # get the layer output of a layer, return an array of layers' names of each layer
    def get_layer_outputs(self):
        return BaseModel.get_layer_outputs(self)


    # return an array of names of all the layers
    def get_all_layer_names(self):
        if self.is_torch:
            all_layer_names = [layer[0] for layer in self.model.named_modules()]
        else:
            all_layer_names = [layer.name for layer in self.model.layers]
        return all_layer_names


    # private method
    def __get_layer_result_by_layer_id(self, x, layer_id, batch_size=None):
        if batch_size is None:
            res = BaseModel.get_layer_result_by_layer_id(self, x, layer_id)
        else:
            r = list()
            n = len(x)
            for i in range(n // batch_size + 1):
                if (i + 1) * batch_size >= n:
                    layer_res = BaseModel.get_layer_result_by_layer_id(self, x[i * batch_size: n], layer_id)
                else:
                    layer_res = BaseModel.get_layer_result_by_layer_id(self, x[i * batch_size: (i + 1) * batch_size], layer_id)

                r.append(layer_res)

                if (i + 1) * batch_size >= n:
                    break

            res = np.concatenate(r, axis=0)

        return res


    # return the layer results of a specific layer by its id
    def get_layer_results_by_layer_id(self, layer_id):
        if self.is_torch:
            return self.__get_layer_result_by_layer_id(self.dataset, layer_id, batch_size=self.batch_size).detach().numpy()
        else:
            return self.__get_layer_result_by_layer_id(self.dataset, layer_id, batch_size=self.batch_size)


    # construct index by layer id, must construct index before doing search if the index has not been constructed or need to search on a different layer_id
    # if the input layer_id has never been constructed, this will construct the index. If the index has been constructed before, it will load the data from memory
    def construct_index(self, layer_id):
        if layer_id in self.index_map.keys():
            self.rev_act = self.index_map[layer_id][0]
            self.rev_idx_act = self.index_map[layer_id][1]
            self.rev_bit_arr = self.index_map[layer_id][2]
            self.rev_idx_idx = self.index_map[layer_id][3]
            self.par_low_bound = self.index_map[layer_id][4]
            self.par_upp_bound = self.index_map[layer_id][5]
        else:
            if self.is_torch:
                self.rev_act, self.rev_idx_act, self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, self.par_upp_bound = helper.construct_index(
                    index_lib=self.index_lib,
                    n_images=self.n_images,
                    ratio=self.ratio,
                    n_partitions=self.n_partitions,
                    bits_per_image=self.bits_per_image,
                    layer_result=self.get_layer_results_by_layer_id(layer_id)
                )
                pack = (self.rev_act, self.rev_idx_act, self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, self.par_upp_bound)
                self.index_map[layer_id] = pack
            else:
                self.rev_act, self.rev_idx_act, self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, self.par_upp_bound = helper.construct_index(
                    index_lib=self.index_lib,
                    n_images=self.n_images,
                    ratio=self.ratio,
                    n_partitions=self.n_partitions,
                    bits_per_image=self.bits_per_image,
                    layer_result=self.get_layer_results_by_layer_id(layer_id)
                )
                pack = (self.rev_act, self.rev_idx_act, self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, self.par_upp_bound)
                self.index_map[layer_id] = pack
        self.current_index_layer_id = layer_id


    # save constructed index_map to disk
    def save_index_map(self, path):
        pickle.dump(self.index_map, open(path, "wb"))

    
    # load saved index_map back to memory
    def load_index_map(self, path):
        self.index_map = pickle.load(open(path, "rb"))


    # private method
    def __get_topk_activations_given_images(self, image_ids, layer_name, k):
        if self.is_torch:
            res = []
            image_samples = []
            for image_sample_id in image_ids:
                image_samples.append(self.dataset[image_sample_id])
            image_samples = torch.stack(image_samples)
            layer_result_image_samples = self.get_layer_result_by_layer_name(image_samples, layer_name)
            for idx, image_sample_id in enumerate(image_ids):
                heap = list()
                for neuron_idx, activation in np.ndenumerate(layer_result_image_samples[idx]):
                    if len(heap) < k:
                        heapq.heappush(heap, (activation, neuron_idx))
                    elif (activation, neuron_idx) > heap[0]:
                        heapq.heapreplace(heap, (activation, neuron_idx))
                temp = sorted(heap, reverse=True)
                res.append(temp)
            return res
        else:
            res = list()
            image_samples = list()
            for image_sample_id in image_ids:
                image_samples.append(self.dataset[image_sample_id])
            layer_result_image_samples = self.get_layer_result_by_layer_name(image_samples, layer_name)
            for idx, image_sample_id in enumerate(image_ids):
                heap = list()
                for neuron_idx, activation in np.ndenumerate(layer_result_image_samples[idx]):
                    if len(heap) < k:
                        heapq.heappush(heap, (activation, neuron_idx))
                    elif (activation, neuron_idx) > heap[0]:
                        heapq.heapreplace(heap, (activation, neuron_idx))
                res.append(sorted(heap, reverse=True))
            return res


    # obtained top k neurons of a given image with specified layer name
    def get_topk_activations_given_images(self, image_ids, layer_name, k):
        topk_activations = self.__get_topk_activations_given_images(image_ids, layer_name, k)
        return topk_activations


    # obtained the nearest k image samples of a certain image, from a specified neuron group 
    def answer_query_with_guarantee(self, layer_id, image_sample_id, k, neuron_group):
        self.construct_index(layer_id)
        if self.is_torch:
            top_k, exit_msg, is_in_partition_0, n_images_rerun = helper.answer_query_with_guarantee(
                                self, self.dataset, self.rev_act, self.rev_idx_act, 
                                self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, 
                                self.par_upp_bound, image_sample_id, neuron_group, k, 
                                self.n_partitions, self.bits_per_image,
                                BATCH_SIZE=self.batch_size_answer, batch_size=self.batch_size_answer)
            top_k = sorted(top_k)
            return top_k, exit_msg
        else:
            top_k, exit_msg, is_in_partition_0, n_images_rerun = helper.answer_query_with_guarantee(
                                self, self.dataset, self.rev_act, self.rev_idx_act, 
                                self.rev_bit_arr, self.rev_idx_idx, self.par_low_bound, 
                                self.par_upp_bound, image_sample_id, neuron_group, k, 
                                self.n_partitions, self.bits_per_image,
                                BATCH_SIZE=self.batch_size_answer, batch_size=self.batch_size_answer)
            top_k = sorted(top_k)
            return top_k, exit_msg
