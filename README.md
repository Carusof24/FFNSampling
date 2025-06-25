# FFNSampling

This repository contains the code to perform samplings of the space of parameters of a generic artificial neural network (ANN).


In the directory models, the class NNModel (see "nn_model.py") takes as input a generic architecture and renders it readable by the classes in other directories (such as 
samplers or  plotting). In models a specific class Feedfoward architecture is presented;

The directory datasets includes datasets classes for handling random-variable data;

The directory samplers contains various classes to perform either Stochastic Gradient Descent training  or canonical ensemble samplings for a given ANN. Each sampler is based on the Hybrid Monte Carlo algorithm for an efficient exploration of the space of parameters.

Utils can be found different functions for file handling and the definition of cost functions and metrics and  to perform operations on the ANN parameters (see "operations.py") and for pseudo-random number generator state file management (see "rng_state.py");
