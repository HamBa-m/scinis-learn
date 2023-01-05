import numpy as np
import numpy.linalg as lg

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
def LinearRegression(X, Y, reg = None, lamda = 0.5):
    '''
    X: matrix of vectors x_i
    Y: vector of labels (scalars, 1 or 0) y_i
    '''
    if reg == "Ridge":
        A = np.dot(X.T,X) + lamda * np.identity(len(X))
    elif reg == "Lasso":
        A = np.dot(X.T,X) 
    elif reg == "net":
        A = np.dot(X.T,X) 
    else :
        A = np.dot(X.T,X) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X.T, Y) 
    # solve the linear system Aplus.b = w
    w = np.dot(lg.inv(A),b)
    return w, loss(X,Y,w)