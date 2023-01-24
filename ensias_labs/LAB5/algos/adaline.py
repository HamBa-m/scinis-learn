import numpy as np

# perceptron hypothesis 
def hs(x,w):
    return w.T @ x

# gradient of loss function
def e(x,y,w):
    return y - w.T @ x

def gradient(X,Y,w):
    return - np.sum([2 * X[i] * e(X[i], Y[i], w) for i in range(len(X))])/len(X)

# empirical loss function
def loss(X,Y,w):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# Single Layer Perceptron with Adaline
def Adaline(X,Y,w, eps = 0.2):
    '''
    X: list of vectors x_i
    Y: list of labels (scalars, 1 or -1) y_i
    w: weights vector
    eps: precision factor
    '''
    n, t = len(X), 0
    lr = 0.00001 # learning rate for the gradient descent
    while abs(gradient(X,Y,w)) > eps and t < 300 :
        print(gradient(X,Y,w))
        for i in range(n):
            if e(X[i], Y[i], w) != 0 : w += 2 * X[i] * e(X[i], Y[i], w) * lr
        print("t=",t," | ",loss(X,Y,w)," | ", gradient(X,Y,w))
        t += 1
    return w, t, loss(X,Y,w)