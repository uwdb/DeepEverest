import numpy as np
import torch.nn

from models.BaseModel_torch import BaseModel_torch


class Mnist_torch(BaseModel_torch):
    # a simple model of mnist 
    # including conv, maxpool, relu, linear, and dropout.

    def __init__(self, model):
        BaseModel_torch.__init__(self, model)

    
    def preprocess_input_for_inference(self, x):
        h, w = 28, 28
        x = np.reshape(x, (-1, h * w))
        numerator = x - np.expand_dims(np.mean(x, 1), 1)
        denominator = np.expand_dims(np.std(x, 1), 1)
        return np.reshape(numerator / (denominator + 1e-7), (-1, h, w, 1))
