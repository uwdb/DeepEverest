# DeepEverest

DNN interpretation is important for researchers and data scientists. While many new approaches are being developed, they often do not scale with the size of the datasets and models. The problem that we address in this paper is the efficient execution of a common class of DNN interpretation queries.

We design, implement, and evaluate DeepEverest, a system for the efficient execution of interpretation by example queries over the activation values of a deep neural network. DeepEverest consists of an efficient indexing technique and a query execution algorithm with various optimizations. Experiments with our prototype implementation show that DeepEverest, using less than 20% of the storage of full materialization, significantly accelerates individual queries by up to 62x and consistently outperforms other methods on multi-query workloads that simulate DNN interpretation processes.

The fundamental building blocks of DNN interpretation are neurons and groups of neurons. To understand what individual neurons and groups of neurons learn and detect, researchers often ask interpretation by example queries. These queries help with understanding the functionality of neurons and neuron groups by tying that functionality to the input examples in the dataset. DeepEverest focuses on accelerating two representative interpretation by example queries: "find the top-k inputs that produce the highest activation values for an individual neuron or group of neurons", and "for any input, find the k-nearest neighbors in the dataset using the activation values of a group of neurons based on the proximity in the latent space defined by the group of neurons".

The preprint version of the paper is available at https://arxiv.org/abs/2104.02234.
