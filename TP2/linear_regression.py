import numpy as np
from numpy import linalg as lg

# linear regression hypothesis 
def hs(x,w):
    return w.T @ x

# cost function
def e(x,y,w):
    return y - hs(x,w)

# empirical error function
def loss(X,Y,w):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# linear regression (algebric approach)
def LinearRegression(X,Y):
    '''
    X: matrix of vectors x_i
    Y: vector of labels (scalars, 1 or 0) y_i
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