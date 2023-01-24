import numpy as np

# perceptron hypothesis 
def hs(x,w):
    """
    description: perceptron hypothesis
    args:
        x: vector of features
        w: vector of weights
    return: matrcial product of x and w.T
    """
    return w.T @ x

# error function between real y and predicted y
def e(x,y,w):
    """
    description: error function between real y and predicted y
    args:
        x: vector of features
        y: label (scalar, 1 or -1)
        w: vector of weights
    return: error between real y and predicted y
    """
    return y - w.T @ x

# empirical loss function
def loss(X,Y,w):
    '''
    description: empirical loss function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    returns: average empirical loss
    '''
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# gradient of the loss function
def gradient(X,Y,w):
    '''
    description: gradient of the loss function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    returns: gradient of the loss function
    '''
    return - np.sum([2 * X[i] * e(X[i], Y[i], w) for i in range(len(X))])/len(X)

# Single Layer Perceptron with Adaline
def Adaline(X,Y,w, delta = 0.2):
    '''
    description: Adaline is a Perceptron with a linear activation function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: weights vector
        eps: precision factor
    returns:
        w: vector of weights after training
        t: number of iterations
        loss: loss function value
    '''
    n, t, Tmax = len(X), 0, 100
    lr = 0.0001 # learning rate for the gradient descent
    while abs(gradient(X,Y,w)) > delta and t < Tmax :
        print(gradient(X,Y,w))
        for i in range(n):
            if e(X[i], Y[i], w) != 0 : w += 2 * X[i] * e(X[i], Y[i], w) * lr
        print("t=",t," | ",loss(X,Y,w)," | ", gradient(X,Y,w))
        t += 1
    return w, t, loss(X,Y,w)