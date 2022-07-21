from abc import ABCMeta, abstractmethod

import numpy as np
from tensorflow.keras import backend as K
import torch
import torch.nn as nn

import torch
import torch.nn as nn

class BaseModel(object):
    __metaclass__ = ABCMeta


    def __init__(self, model, is_torch, optimizer=None):
        self.is_torch = is_torch

        if self.is_torch:
            self.model = model
            self.model_dict = dict(model.named_modules())
            self.name_list = []
            self.get_layer_outputs_ = {}
            def get_activation(name):
                def hook(model, input, output):
                    self.get_layer_outputs_[name] = output.detach()
                return hook
            for name, module in model.named_modules():
                self.name_list.append(name)
                self.get_layer_outputs_[name] = module.register_forward_hook(get_activation(name))

        else:
            self.model = model
            self.optimizer = optimizer
            self.get_layer_outputs_ = [K.function([self.model.layers[0].input], [self.model.layers[layer_id].output]) for
                                    layer_id in range(len(self.model.layers))]


    def get_layer_outputs(self):
        return self.get_layer_outputs_


    def get_model(self):
        return self.model


    def load_weights(self, path):
        if self.is_torch:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        else:
            self.model.load_weights(path)


    def save(self, path):
        if self.is_torch:
            torch.save(self.model.state_dict(), path)
        else:
            self.model.save(path)


    def get_layer_result_by_layer_id(self, x, layer_id, normalize=True):
        if self.is_torch:

            layer_name = self.name_list[layer_id]
            output = self.model(x)
            result = self.get_layer_outputs_[layer_name]
            return result

        else:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if normalize:
                x = self.preprocess_input_for_inference(x)
            get_layer_output = self.get_layer_outputs_[layer_id]
            res = get_layer_output(x)[0]
            return res


    def get_layer_result_by_layer_name(self, x, layer_name, normalize=True):
        if self.is_torch:
            output = self.model(x)
            result = self.get_layer_outputs_[layer_name]
            return result
        else:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if normalize:
                x = self.preprocess_input_for_inference(x)
            layer_id = [layer.name for layer in self.model.layers].index(layer_name)
            get_layer_output = self.get_layer_outputs_[layer_id]
            res = get_layer_output(x)[0]
            return res


    def get_layer_results_by_layer_names(self, x, layer_names, normalize=True):
        if self.is_torch:
            output = self.model(x)
            result = []
            for name in layer_names:
                result.append(self.get_layer_outputs_[name])
            return result
        else:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if normalize:
                x = self.preprocess_input_for_inference(x)
            get_layer_output = K.function([self.model.layers[0].input],
                                        [self.model.get_layer(layer_name).output for layer_name in layer_names])
            res = get_layer_output(x)
            return res


    @abstractmethod
    def preprocess_input_for_inference(self, x):
        raise NotImplementedError("Must override preprocess_input_for_inference")
