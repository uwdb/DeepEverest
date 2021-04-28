import numpy as np


class NeuronGroup:
    def __init__(self, model, layer_id, dimension_ranges=None, neuron_idx_list=None):
        self.layer_id = layer_id
        if neuron_idx_list is not None:
            self.neuron_idx_list = neuron_idx_list
        else:
            self.neuron_idx_list = list()
            output_shape = model.layers[layer_id].output_shape[1:]
            for neuron_idx in np.ndindex(output_shape):
                if dimension_ranges is not None:
                    if not self.neuron_in_range(dimension_ranges, neuron_idx):
                        continue
                self.neuron_idx_list.append(neuron_idx)

    @staticmethod
    def neuron_in_range(dimension_ranges, neuron_idx):
        for dimension_idx, dimension_range in enumerate(dimension_ranges):
            if dimension_range[0] <= neuron_idx[dimension_idx] < dimension_range[1]:
                pass
            else:
                return False
        return True
