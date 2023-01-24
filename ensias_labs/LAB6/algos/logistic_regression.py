import numpy as np
from numpy import linalg as lg

# activation function (sigmoid)
def phi(x):
    """
    description: activation function (sigmoid)
    args:
        x: scalar
    return: sigmoid of x
    """
    return 1 / (1 + np.exp(-x))

# logistic regression hypothesis
def hs(x,w):
    """
    description: logistic regression hypothesis
    args:
        x: vector of features
        w: vector of weights
    return: matrcial product of x and w.T
    """
    return w.T @ x 

# cross-entropy error function
def ls(x,y,w):
    """
    description: cross-entropy error function
    args:
        x: vector of features
        y: label (scalar, 1 or -1)
        w: vector of weights
    return: cross-entropy error between real y and predicted y
    """
    return - np.log(phi(y * hs(x,w)))

# empirical loss function
def loss(X,Y,w):
    '''
    description: empirical loss function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    return: average empirical loss
    '''
    n = len(X) # size of data sample
    error = [ls(X[i], Y[i], w) for i in range(len(X))]
    return np.sum(error)/n

# gradient of loss function
def gradient(X,Y,w):
    '''
    description: gradient of loss function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    return: gradient of loss function
    '''
    n = len(X) # size of data sample
    d = len(w) # dimension of data
    dL = np.zeros(d) # initialize gradient vector
    for i in range(n): 
        dL += phi(- Y[i] * hs(X[i], w)) * (- Y[i] * X[i]) 
    return dL / n

# logistic regression
def LogisticRegression(X,Y, lr = 0.1, Tmax = 2000, epsilon = 0.4):
    '''
    description: logistic regression with gradient descent
    args:
        X: matrix of vectors x_i
        Y: vector of labels (scalars, 1 or 0) y_i
        lr: learning rate
        Tmax: maximum number of iterations
        epsilon: threshold for stopping criterion (precision factor)
    return: 
        w: vector of weights after training
        t: number of iterations
        Ls: empirical loss
    '''
    t = 0 # iteration counter
    w = np.zeros(X.shape[1]) # initialize weights vector
    Ls = loss(X,Y,w) # empirical loss
    while(np.linalg.norm(Ls) > epsilon and t < Tmax): # stopping criterion
        print("iter:",t,"\t| empirical loss: ", "{0:.6f}".format(Ls)) # print loss
        w -= lr * gradient(X,Y,w) # gradient descent update
        Ls = loss(X,Y,w) # empirical loss update
        t += 1 # iteration counter update
    return w, t, Ls