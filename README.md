# DeepEverest: A System for Efficient DNN Interpretation

Prototype implementation of DeepEverest, which is a system that supports efficient DNN *interpretation by example* queries. The paper will appear in PVLDB Vol. 15 (coming soon!), and the preprint version of the paper is available at https://arxiv.org/abs/2104.02234.

# Cloning
`git clone https://github.com/uwdb/DeepEverest.git`
`cd DeepEverest`

# Install required packages
The prototype is tested with Python 3.7. You can enter your virtual environment before this step.

`pip install -r requirements.txt`

# Example usage

## Build the dynamically linked libraries for index construction in DeepEverest.
`cd index`
`python setup_deepeverest_index.py build`

You should be able to see a `build` folder in your current directory. One of the directories (directory name depending on system and python version) inside `build` will contain the built libraries. It is a `.so` file. The filename is also dependent on the system and python versions.


`python 3`

```

```

# Running the example notebook
You can run `example.ipynb` to walk through the functionality that DeepEverest provides.
