import numpy as np
import numpy.linalg as lg
import sys
import colorama

# linear regression hypothesis 
def hs(x,w):
    return w.T @ x

# cost function
def e(x,y,w):
    return y - hs(x,w)

# loss function
def loss(X,Y,w, reg=None, lamda=0.5, alpha = 0.5):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    if reg == "Ridge" : 
        return np.sum(error)/n + lamda * np.sum(np.power(w,2))
    elif reg == "Lasso" :
        return np.sum(error)/n + lamda * np.sum(np.array([abs(e) for e in w]))
    elif reg == "Elastic" :
        return np.sum(error)/(2*n) + lamda * (1 - alpha)/2 * np.sum(np.power(w,2)) + lamda * alpha * np.sum(np.array([abs(e) for e in w]))
    return np.sum(error)/n

# gradient of loss function
def gradient(X,Y,w, reg=None, lamda = 0.5, alpha = 0.5):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w)) for i in range(len(X))]
    if reg == "Ridge" : 
        return (2/n) * np.dot(X.T,error) + lamda * 2 * w
    elif reg == "Lasso" :
        return (2/n) * np.dot(X.T,error) + lamda * np.sign(w) # using sub-gradient
    elif reg == "Elastic" :
        return (1/n) * np.dot(X.T,error) + lamda * (1 - alpha) * w + lamda * alpha * np.sign(w) 
    return (2/n) * np.dot(X.T,error)

# linear regression (with regularization)
def LinearRegression(X, Y, reg = None, lamda = 0.5, alpha = 0.5, lr = 0.01):
    '''
    X: matrix of vectors x_i
    Y: vector of labels (scalars, 1 or 0) y_i
    reg: type of regularization, either Ridge, Lasso, Elastic Net, or None
    lamda: parameter of regularization
    alpha: parameter of Elastic Net regularization
    lr: learning rate of the gradient descent
    '''
    w = np.zeros(X.shape[1]) # initialize weights vector

    if reg == "Ridge": # Ridge case could be done fast with an algebric approach
        I = np.identity(len(X.T))
        I[0][0] = 0
        A = np.dot(X.T,X) + lamda * I
        # computes the second term of the linear system A.w = b
        b = np.dot(X.T, Y) 
        # solve the linear system Aplus.b = w
        w = np.dot(lg.inv(A),b)

    elif reg != None: # Lasso and Elastic Net cases needs an iterative search using gradient descent
        Tmax = 200
        for i in range(Tmax):
            dw = gradient(X, Y, w, reg=reg, lamda=lamda, alpha=alpha)
            w += lr * dw 
            sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((i+1)/Tmax))+"-" *int(50 * (1 - (i+1)/Tmax)),int(100*(i+1)/Tmax)))
            sys.stdout.flush()
        sys.stdout.write(colorama.Fore.WHITE)

    else : # simple linear regression
        A = np.dot(X.T,X) 
        b = np.dot(X.T, Y)
        w = np.dot(lg.pinv(A),b)

    return w, loss(X,Y,w, reg=reg, lamda=lamda,alpha=alpha)