# DeepEverest: A System for Efficient DNN Interpretation

A prototype implementation of DeepEverest, which is a system that supports efficient DNN *interpretation by example* queries. DeepEverest focuses on accelerating <i>interpretation by example</i> queries that return inputs (e.g., images) in the dataset that have certain neuron activation patterns, e.g., "given a group of neurons, find the top-k inputs that produce the highest activation values for this group of neurons", and "for any input and any group of neurons, use the activations of the neurons to identify the nearest neighbors based on the proximity in the space learned by the neurons". These queries help with understanding the functionality of neurons and neuron groups by tying that functionality to the input examples in the dataset.  We provide instructions on how you can apply DeepEverest to your own models and datasets [below](#ownmodel).

See [project website](https://db.cs.washington.edu/projects/deepeverest/) for more details. A [paper](http://vldb.org/pvldb/vol15/p98-he.pdf) for this project is published in PVLDB Vol. 15, [doi:10.14778/3485450.3485460](https://doi.org/10.14778/3485450.3485460). An [extended technical report](https://arxiv.org/abs/2104.02234) is also available.

## Repository Overview
An example notebook is `example.ipynb`. Implementations of core functionalities of DeepEverest are in `DeepEverest.py` and `index/deepeverst_index.cpp`. The DNN models and datasets used in the paper are in `models/`. However, you can apply DeepEverest on your own model and dataset. `index/` contains the core source for the construction of the indexes used in the DeepEverest. `tools/` contains useful interpretation techniques adapted from other projects. `utils.py` contains frequently used functions.

## Cloning
**Install [Git Large File Storage](https://git-lfs.github.com/) before cloning** the repository, then,

`git clone git@github.com:uwdb/DeepEverest.git` (or `git clone https://github.com/uwdb/DeepEverest.git` when ssh does not work) <br>

**Clone model weights** tracked by `git lfs` by,

`cd DeepEverest` <br>
`git lfs install` <br>
`git lfs pull`

## Installation of required packages
The prototype is tested with Python 3.7. You can enter your virtual environment before this step.

`pip install -r requirements.txt`

## Example usage

### Build the dynamically linked library for the construction of the Neural Partition Index (NPI) and Maximum Activation Index (MAI) in DeepEverest
`cd index` <br>
`python setup_deepeverest_index.py build`

You should be able to see a `build` folder in your current directory. One of the directories (directory name depending on system and python versions) inside `build` will contain the built library. It is a `.so` file. The filename is also dependent on the system and python versions. For example, the relative path could look like `index/build/lib.macosx-10.7-x86_64-3.7/deepeverst_index.cpython-37m-darwin.so` if you build the library on a MacOS with an Intel CPU using Python 3.7.


### Construct the indexes (NPI and MAI)
Go to the root directory of DeepEverest.

`python 3`

```
# Load the built library.
import ctypes

lib_file = <the path of the .so file that you just built>
index_lib = ctypes.CDLL(lib_file)

# Load the model and dataset that you want to interpret. Note that you can load your own model and dataset.
from utils import load_mnist_vgg_dataset_model

x_train, y_train, x_test, y_test, model = load_mnist_vgg_dataset_model()
all_layer_names = [layer.name for layer in model.model.layers]
dataset = x_test

# Set the layer of interest and get its activations by running DNN inference.
from utils import get_layer_result_by_layer_id

layer_name = "activation_12"
layer_id = all_layer_names.index(layer_name)
batch_size = 64
layer_result = get_layer_result_by_layer_id(model, dataset, layer_id, batch_size=batch_size)

# Configure the parameters of the indexes to be built. Note that you can set your own configuration.
n_images = len(dataset)
n_partitions= 32
ratio = 0.05
import math
bits_per_image = math.ceil(math.log(n_partitions, 2))

# Construct the indexes.
from DeepEverest import construct_index

rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx, par_l_bnd, par_u_bnd = construct_index(
                                                                        index_lib=index_lib,
                                                                        n_images=n_images,
                                                                        ratio=ratio,
                                                                        n_partitions=n_partitions,
                                                                        bits_per_image=bits_per_image,
                                                                        layer_result=layer_result)

```

You can choose to persist the indexes to disk with `np.save()` or `pickle.dump()` to accelerate future interpretation for this layer, or to interpret your DNN and dataset directly.

### Interpret the functionalities of any group of neurons using DeepEverest's Neural Threshold Algorithm (NTA)

```
# Set the target input of interest and the number of top activations you want to inspect.
# For example, image 659 is a misclassified example in the dataset.
image_ids = [659]
n_neurons = 5

# Get the top-k activations for this input in this layer and their corresponding neuron IDs.
from utils import get_topk_activations_given_images

topk_activations = get_topk_activations_given_images(model, dataset, image_ids, layer_name, n_neurons)[0]
topk_activations_neurons = [x[1] for x in topk_activations]

# Construct the group of neurons that you are interested in, e.g., the top-3 maximally activated neurons.
from NeuronGroup import *

image_sample_id = image_ids[0]
neuron_group = NeuronGroup(model.model, layer_id, neuron_idx_list=topk_activations_neurons[:3])

# Query for the k-nearest neighbors in the dataset using the activations of this group of neurons
# based on the proximity in the latent space defined by this group of neurons.
# answer_query_with_guarantee() runs the Neural Threshold Algorithm.
from DeepEverest import answer_query_with_guarantee

k = 20
top_k, exit_msg, _, n_images_run = answer_query_with_guarantee(
                                    model, dataset, rev_act, rev_idx_act, rev_bit_arr, rev_idx_idx,
                                    par_l_bnd, par_u_bnd, image_sample_id, neuron_group, k,
                                    n_partitions, bits_per_image, BATCH_SIZE=batch_size, batch_size=batch_size)
                                    

# Sort the top-k results based on their negative distances to the target input.
top_k = sorted(top_k)

# Visualize the top-k results.
from utils import plot_mnist

for neg_dist, image_id in top_k:
    plot_mnist(x_test, label_test, image_id)
```

The top-k results in `top_k`. Inspect them to investigate and understand the group of neurons' functionality by tying that functionality to the input examples in the dataset.

## Running the example notebook
You can run `example.ipynb` to walk through the functionalities that DeepEverest provides. `old-examples/` also contains a few more examples for an old version of DeepEverest with some other useful interpretation techniques adapted from other projects (e.g., pixel-level attribution), which probably only works with Tensorflow 1.x.

## Working with your own model <a name="ownmodel"></a>
To apply DeepEverest on your own raw model (currently supporting `tf.keras` models), create a subclass of `BaseModel` in `models/` because DeepEverest relies on methods of `BaseModel`. For example, create a file `CustomModel.py` in `models/`,

```
from models.BaseModel import BaseModel

class CustomModel(BaseModel):
    def __init__(self, model):
        BaseModel.__init__(self, model=model, optimizer=None)
    def preprocess_input_for_inference(self, x):
        return x
```

In your main script, load your own raw model and wrap it in `CustomModel` so that DeepEverest can work.

```
from tensorflow.keras.models import load_model
from models.CustomModel import CustomModel

raw_model = load_model('your_own_model.h5')
model = CustomModel(raw_model)
```

Yay, now you can use DeepEverest with `model`.

## Citations & Paper

If you find DeepEverest useful, please cite our paper:

_DeepEverest: Accelerating Declarative Top-K Queries for Deep Neural Network Interpretation_<br />
Dong He, Maureen Daum, Walter Cai, Magdalena Balazinska<br />
Proc. VLDB Endow. 15(1): 98-111 (2021) [[PDF](https://doi.org/10.14778/3485450.3485460)]

```
@article{DBLP:journals/pvldb/HeDCB21,
  author    = {Dong He and
               Maureen Daum and
               Walter Cai and
               Magdalena Balazinska},
  title     = {DeepEverest: Accelerating Declarative Top-K Queries for Deep Neural
               Network Interpretation},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {15},
  number    = {1},
  pages     = {98--111},
  year      = {2021},
  url       = {http://www.vldb.org/pvldb/vol15/p98-he.pdf},
  biburl    = {https://dblp.org/rec/journals/pvldb/HeDCB21.bib}
}
```

See the [project website](https://db.cs.washington.edu/projects/deepeverest/) for more details about DeepEverest.

