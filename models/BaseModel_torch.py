from abc import ABCMeta, abstractmethod
from doctest import OutputChecker
from matplotlib.pyplot import get

import numpy as np
import torch
import torch.nn as nn


class BaseModel_torch(object):
    __metaclass__ = ABCMeta

    def __init__(self, model):
        self.model = model

        self.model_dict = dict(model.named_modules())
        self.name_list = []
        self.get_layer_outputs = {}
        def get_activation(name):
            def hook(model, input, output):
                self.get_layer_outputs[name] = output.detach()
            return hook
        for name, module in model.named_modules():
            self.name_list.append(name)
            self.get_layer_outputs[name] = module.register_forward_hook(get_activation(name))
            
    
    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()


    def save(self, path):
        torch.save(self.model.state_dict(), path)

    # def compile(self):
    #     self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # not changed
    # def fit(self, training_data, validation_data, epochs, batch_size):
    #     x_train, y_train = training_data
    #     x_val, y_val = validation_data

    #     hist = self.model.fit(x_train, y_train, epochs=epochs,
    #                           batch_size=batch_size,
    #                           validation_data=(x_val, y_val), callbacks=self.callbacks)
    #     return hist

    # not changed
    # def evaluate(self, eval_data, batch_size=32):
    #     x, y = eval_data
    #     loss_and_metrics = self.model.evaluate(x, y,
    #                                            batch_size=batch_size)
    #     return loss_and_metrics

    def predict(self, x, normalize=False, batch_size=500):
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if normalize:
            x = self.preprocess_input_for_inference(x)
        return self.model(x)


    def get_layer_result_by_layer_id(self, x, layer_id, normalize=False):
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if normalize:
            x = self.preprocess_input_for_inference(x)
        layer_name = self.name_list[layer_id]
        output = self.model(x)
        return self.get_layer_outputs[layer_name]


    def get_layer_result_by_layer_name(self, x, layer_name, normalize=False):
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if normalize:
            x = self.preprocess_input_for_inference(x)
        output = self.model(x)
        return self.get_layer_outputs[layer_name]


        #layer_id = [layer.name for layer in self.model.layers].index(layer_name)
        #get_layer_output = self.get_layer_outputs[layer_id]
        #res = get_layer_output(x)[0]

    def get_layer_results_by_layer_names(self, x, layer_names, normalize=False):
        # if not torch.is_tensor(x):
        #     x = torch.tensor(x)
        if normalize:
            x = self.preprocess_input_for_inference(x)
        output = self.model(x)
        result = []
        for name in layer_names:
            result.append(self.get_layer_outputs[name])
        return result

        
        # get_layer_output = K.function([self.model.layers[0].input],
        #                               [self.model.get_layer(layer_name).output for layer_name in layer_names])
        # res = get_layer_output(x)
        # return res

    @abstractmethod
    def preprocess_input_for_inference(self, x):
        raise NotImplementedError("Must override preprocess_input_for_inference")
