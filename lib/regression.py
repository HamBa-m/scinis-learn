import numpy as np
from numpy import linalg as lg
from .loss import lossAM, DlossAM
import colorama # for colored output in terminal
import sys

################ regression ERM algorithms ################
# linear regression with regularization variants
def LinearRegression(X, Y, reg = None, lamda = 0.5, alpha = 0.5, lr = 0.01):
    '''
    description: linear regression with regularization variants
        X: matrix of vectors x_i
        Y: vector of labels (scalars, 1 or 0) y_i
        reg: type of regularization, either Ridge, Lasso, Elastic Net, or None
        lamda: parameter of regularization
        alpha: parameter of Elastic Net regularization
        lr: learning rate of the gradient descent
    return: weight vector w and loss function
    '''
    w = np.zeros(X.shape[1]) # initialize weights vector

    if reg == "Ridge": # Ridge case could be done fast with an algebraic approach
        I = np.identity(len(X.T))
        I[0][0] = 0
        A = np.dot(X.T,X) + lamda * I
        # computes the second term of the linear system A.w = b
        b = np.dot(X.T, Y) 
        # solve the linear system Aplus.b = w
        w = np.dot(lg.inv(A),b)

    elif reg != None: # Lasso and Elastic Net cases needs an iterative search using (sub)gradient descent
        Tmax = 200 # upper bound on number of iterations 
        for i in range(Tmax):
            dw = DlossAM(X, Y, w, reg=reg, lamda=lamda, alpha=alpha) # compute gradient
            w -= lr * dw  # update weights 
            sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((i+1)/Tmax))+"-" *int(50 * (1 - (i+1)/Tmax)),int(100*(i+1)/Tmax))) 
            # progress bar
            sys.stdout.flush() 
        sys.stdout.write(colorama.Fore.WHITE) # reset color of terminal

    else : # simple linear regression case (no regularization)
        A = np.dot(X.T,X) 
        b = np.dot(X.T, Y)
        w = np.dot(lg.pinv(A),b)

    return w, lossAM(X,Y,w, reg=reg, lamda=lamda,alpha=alpha)