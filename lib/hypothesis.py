import numpy as np

################ hypothesis function ################
def h(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    returns: scalar value of the hypothesis
    """
    return w.T @ x