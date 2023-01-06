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
def LogisticRegression(X,Y, lr = 0.1, Tmax = 2000, epsilon = 0.4):
    t = 0 #initialisation de compteur
    w = np.zeros(X.shape[1])
    Ls = loss(X,Y,w)
    while(np.linalg.norm(Ls) > epsilon and t < Tmax):
        print("iter:",t,"\t| empirical loss: ", "{0:.6f}".format(Ls))
        w -= lr * gradient(X,Y,w) # gradient descent update
        Ls = loss(X,Y,w)
        t += 1
    return w, t, np.linalg.norm(Ls)