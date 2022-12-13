import numpy as np
from numpy import linalg as lg

# activation function (sigmoid)
def phi(x):
    return 1 / (1 + np.exp(-x))

# logistic regression hypothesis
def hs(x,w):
    return w.T @ x 

# cross-entropy error function
def ls(x,y,w):
    return - np.log(phi(y * hs(x,w)))

# empirical loss function
def loss(X,Y,w):
    n = len(X) # size of data sample
    error = [ls(X[i], Y[i], w) for i in range(len(X))]
    return np.sum(error)/n

# gradient of loss function
def gradient(X,Y,w):
    n = len(X)
    d = len(w)
    dL = np.zeros(d)
    for i in range(n):
        dL += phi(- Y[i] * hs(X[i], w)) * (- Y[i] * X[i])
    return dL / n

# logistic regression
def LogisticRegression(X,Y, lr = 0.5, Tmax = 1000):
    '''
    X: matrix of vectors x_i
    Y: vector of labels (scalars, 1 or 0) y_i
    lr: learning rate for the gradient descent, 0.5 by default
    Tmax: maximum number of iteration
    '''
    w = np.array([0.]*len(X[0]))
    n, t = len(X), 0
    while lg.norm(gradient(X,Y,w)) != 0 and t < Tmax :
        w -= gradient(X, Y, w) * lr
        print("t=",t,"\t| empirical loss: ", "{0:.6f}".format(loss(X,Y,w)))
        t += 1
    return w, t, loss(X,Y,w)