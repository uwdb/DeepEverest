# DeepEverest: A System for Efficient DNN Interpretation

Prototype implementation of DeepEverest, which is a system that supports efficient DNN *interpretation by example* queries. The paper will appear in PVLDB Vol. 15 (coming soon!), and the preprint version of the paper is available at https://arxiv.org/abs/2104.02234.

## Cloning
`git clone https://github.com/uwdb/DeepEverest.git` <br>
`cd DeepEverest`

## Install required packages
The prototype is tested with Python 3.7. You can enter your virtual environment before this step.

`pip install -r requirements.txt`

## Example usage

### Build the dynamically linked library for index construction in DeepEverest.
`cd index` <br>
`python setup_deepeverest_index.py build`

You should be able to see a `build` folder in your current directory. One of the directories (directory name depending on system and python version) inside `build` will contain the built library. It is a `.so` file. The filename is also dependent on the system and python versions.


### Build the indexes.
`python 3`

```
# Load the built library
import ctypes
lib_file = <the path of the .so file that you just built>
index_lib = ctypes.CDLL(lib_file)

# Load the model and dataset that you want to interpret
from utils import load_mnist_vgg_dataset_model
x_train, y_train, x_test, y_test, model = load_mnist_vgg_dataset_model()
all_layer_names = [layer.name for layer in model.model.layers]
dataset = x_test

# Set the layer of interest and get its activations
layer_name = "activation_12"
layer_id = all_layer_names.index(layer_name)
batch_size = 64
layer_result = get_layer_result_by_layer_id(model, dataset, layer_id, batch_size=batch_size)

# Configure the indexes to be built
n_images = len(dataset)
n_partitions= 32
ratio = 0.05
import math
bits_per_image = math.ceil(math.log(n_partitions, 2))

# Build the indexes
from DeepEverest import *
rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx, par_low_bound, par_upp_bound = construct_index(
  index_lib=index_lib,
  n_images=n_images,
  ratio=ratio,
  n_partitions=n_partitions,
  bits_per_image=bits_per_image,
  layer_result=layer_result)

```

You can choose to persist the indexes to disk with `np.save()` or `pickle.dump()`, or to interpret your DNN and dataset directly.

### Interpret the functionality of any group of neurons.

```
# Set the target input of interest and the number of top activations you want to inspect
image_ids = [659]
k = 20

# Get the top-k activations for this input in this layer and their corresponding neuron ids
from utils import get_topk_activations_given_images
topk_activations = get_topk_activations_given_images(model, dataset, image_ids, layer_name, k_global)[0]
topk_activations_neurons = [x[1] for x in topk_activations]

# Construct the group of neurons that you are interested in, e.g., the top-3 maximally activated neurons
from NeuronGroup import *
image_sample_id = 659
neuron_group = NeuronGroup(model.model, layer_id, neuron_idx_list=topk_activations_neurons[:3])

# Query for the k-nearest neighbors in the dataset using the activations of this group of neurons based on the proximity in the latent space defined by this group of neurons.
top_k, exit_msg, _, n_images_run = answer_query_with_guarantee(
                                                        model, dataset, rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx,
                                                        par_low_bound, par_upp_bound, image_sample_id,
                                                        neuron_group, k_global, n_partitions, bits_per_image,
                                                        BATCH_SIZE=batch_size, batch_size=batch_size)
top_k = sorted(top_k)
```

The top-k results in `top_k`. Inspect them to investigate and understand what this group of neurons' functionality by tying that functionality to the input examples in the dataset.

# Running the example notebook
You can run `example.ipynb` to walk through the functionality that DeepEverest provides.
