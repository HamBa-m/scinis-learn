import numpy as np
from .hypothesis import h

################ activation functions ################
# sign function
def sign(x,y,w):
    """
    args:
        x: vector of features
        w: vector of weights
    return: 1 if h(x,w) > 0, -1 otherwise
    """
    if np.sign(h(x,w)) * y > 0 : return 1
    return -1

# sigmoid function
def sigmoid(x):
    """
    description: activation function (sigmoid)
    args:
        x: scalar
    return: sigmoid of x
    """
    return 1 / (1 + np.exp(-x))