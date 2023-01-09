import numpy as np
from numpy import linalg as lg

# linear regression hypothesis 
def hs(x,w):
    """
    description: linear regression hypothesis
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
    return y - hs(x,w)

# empirical error function
def loss(X,Y,w):
    '''
    description: empirical error function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    returns: average empirical error   
    '''
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# linear regression (algebric approach)
def LinearRegression(X,Y):
    '''
    description: linear regression (algebric approach)
    args:
        X: matrix of vectors x_i
        Y: vector of labels (scalars, 1 or 0) y_i
    returns:
        w: vector of weights
        loss: empirical loss
    '''
    # computes the Hessian matrix of the loss function applied to the hypothesis
    A = np.dot(X.T,X) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X.T, Y) 
    # computes the pseudoinverse of A using a Singular-Value Decomposition algorithm
    Aplus = lg.pinv(A)
    # solve the linear system Aplus.b = w
    w = np.dot(Aplus,b)
    return w, loss(X,Y,w)