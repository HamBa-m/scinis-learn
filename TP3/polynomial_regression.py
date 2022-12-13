import numpy as np
import numpy.linalg as lg

# polynomial regression mapping
def phi(x, n):
    """
    x: the entry variable
    n: the degree of the output polynome
    """
    return np.array([x**i for i in range(n+1)]).T

# polynomial regression hypothesis 
def hs(x,w):
    return w.T @ x

# gradient of loss function
def e(x,y,w):
    return y - w.T @ x

# empirical loss function
def loss(X,Y,w):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# Polynomial regression for polynomial tasks
def PolynmialRegression(X,Y,n):
    '''
    X: matrix of vectors x_i
    Y: vector of labels (scalars, 1 or 0) y_i
    n: degree of polynomial mapping
    '''
    # mapping the dataset to an upper degree
    X_ = np.asarray([phi(x,n) for x in X])
    # computes the Hessian matrix of the loss function applied to the hypothesis
    A = np.dot(X_.T,X_) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X_.T, Y) 
    # computes the pseudoinverse of A using a Singular-Value Decomposition algorithm
    Aplus = np.linalg.pinv(A)
    # solve the linear system A+.w = b
    w = np.dot(Aplus,b)
    return w, loss(X_,Y,w)