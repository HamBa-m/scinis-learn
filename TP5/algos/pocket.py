import numpy as np

#perceptron hypothesis 
def hs(x,w):
    if np.sign(w.T @ x) > 0 : return 1
    return -1

# empirical loss function
def loss(X,Y,w):
    n = len(X) # size of data sample
    misclassified = [1 if hs(X[i], w) != Y[i] else 0 for i in range(len(X))]
    return sum(misclassified)/n

# Single Layer Perceptron with Pocket
def Pocket(X,Y,w):
    '''
    X: list of vectors x_i
    Y: list of labels (scalars, 1 or 0) y_i
    '''
    n, t = len(X), 0
    Tmax = 300
    w0 = np.array(w)
    while t < Tmax and loss(X,Y,w) > 0.01:
        for i in range(n):
            if hs(X[i], w0) * Y[i] < 0 : w0 += X[i]*Y[i]
        t += 1
        print("t=",t," | ",loss(X,Y,w))
        if loss(X,Y,w0) < loss(X,Y,w) : w = w0
    return w, t, loss(X,Y,w)

    