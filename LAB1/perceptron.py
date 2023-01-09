import numpy as np

# pereceptron hypothesis
def h(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    returns: scalar value of the hypothesis
    """
    return w.T @ x

# hypothesis sign
def hs(x,w):
    """
    args:
        x: vector of features
        w: vector of weights
    return: 1 if h(x,w) > 0, -1 otherwise
    """
    if np.sign(h(x,w)) > 0 : return 1
    return -1

# empirical loss function
def loss(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: average empirical loss
    '''
    n = len(X) # size of data sample
    misclassified = [1 if hs(X[i], w) != Y[i] else 0 for i in range(len(X))]
    return sum(misclassified)/n

# Single Layer Perceptron
def PLA(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: 
        w: vector of weights after training
        t: number of iterations
    '''
    n, t = len(X), 0
    while loss(X,Y,w) != 0:
        for i in range(n):
            if hs(X[i], w) * Y[i] < 0 : w += X[i]*Y[i]
        t += 1
        print("t=",t," | ",loss(X,Y,w))
    return w, t