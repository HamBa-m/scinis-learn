import numpy as np
import numpy.linalg as lg
import sys
import colorama # for colored output in terminal 

# linear regression hypothesis 
def hs(x,w):
    return w.T @ x

# cost function
def e(x,y,w):
    return y - hs(x,w)

# loss function with regularization variants
def loss(X,Y,w, reg=None, lamda=0.5, alpha = 0.5):
    """
    description: loss function with regularization variants
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
        reg: regularization variant (Ridge, Lasso, Elastic, None)
        lamda: regularization parameter
        alpha: ElasticNet mixing parameter
    return: loss function with regularization variants
    """
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))] # squared error vector
    if reg == "Ridge" : 
        return np.sum(error)/n + lamda * np.sum(np.power(w,2)) 
    elif reg == "Lasso" :
        return np.sum(error)/n + lamda * np.sum(np.array([abs(e) for e in w])) 
    elif reg == "Elastic" :
        return np.sum(error)/(2*n) + lamda * (1 - alpha)/2 * np.sum(np.power(w,2)) + lamda * alpha * np.sum(np.array([abs(e) for e in w]))
    return np.sum(error)/n

# gradient of loss function
def gradientLin(X,Y,w, reg=None, lamda = 0.5, alpha = 0.5):
    """
    description: gradient of loss function
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
        reg: regularization variant (Ridge, Lasso, Elastic, None)
        lamda: regularization parameter
        alpha: ElasticNet mixing parameter
    return: loss function with regularization variants
    """
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w)) for i in range(len(X))] # error vector 
    if reg == "Ridge" : 
        return (2/n) * np.dot(X.T,error) + lamda * 2 * w # using gradient
    elif reg == "Lasso" :
        return (2/n) * np.dot(X.T,error) + lamda * np.sign(w) # using sub-gradient
    elif reg == "Elastic" :
        return (1/n) * np.dot(X.T,error) + lamda * (1 - alpha) * w + lamda * alpha * np.sign(w) 
    return (2/n) * np.dot(X.T,error)

# linear regression (with regularization)
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
            dw = gradientLin(X, Y, w, reg=reg, lamda=lamda, alpha=alpha) # compute gradient
            w += lr * dw  # update weights 
            sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((i+1)/Tmax))+"-" *int(50 * (1 - (i+1)/Tmax)),int(100*(i+1)/Tmax))) 
            # progress bar
            sys.stdout.flush() 
        sys.stdout.write(colorama.Fore.WHITE) # reset color of terminal

    else : # simple linear regression case (no regularization)
        A = np.dot(X.T,X) 
        b = np.dot(X.T, Y)
        w = np.dot(lg.pinv(A),b)

    return w, loss(X,Y,w, reg=reg, lamda=lamda,alpha=alpha)


#################################################################################
# activation function (sigmoid)
def phi(x):
    return 1 / (1 + np.exp(-x))

# cross-entropy error function
def ls(x,y,w):
    return - y* np.log(phi(hs(x,w))) - (1 - y) * np.log(1 - phi(hs(x,w)))

# loss function for logistic regression (with regularization)
def loss(X,Y,w, reg=None, lamda=0.5, alpha = 0.5):
    """
    description: loss function with regularization variants (Ridge, Lasso, Elastic, None)
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
        reg: regularization variant (Ridge, Lasso, Elastic, None)
        lamda: regularization parameter
        alpha: ElasticNet mixing parameter
    return: loss function with regularization variants
    """
    n = len(X) # size of data sample
    error = [ls(X[i], Y[i], w) for i in range(len(X))]

    if reg == "Elastic":
        return np.sum(error)/n + (lamda * alpha) * np.sum(np.array([abs(e) for e in w]))+ (lamda * (1 - alpha) /(2 *n)) * np.sum(np.power(w,2))
    elif reg == "Ridge" : 
        return np.sum(error)/n + (lamda/(2 *n)) * np.sum(np.array([abs(e) for e in w]))
    elif reg == "Sparse" :
        return np.sum(error)/n + (lamda/(2 *n)) * np.sum(np.power(w,2))

    return np.sum(error)/n

# gradient of loss function
def gradientReg(X,Y,w, reg=None, lamda = 0.5, alpha = 0.5):
    """
    description: gradient of loss function with regularization variants (Ridge, Sparse, Elastic, None)
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
        reg: regularization variant (Ridge, Sparse, Elastic, None)
        lamda: regularization parameter
        alpha: ElasticNet mixing parameter
    return: gradient of loss function with regularization variants
    """
    n = len(X) # size of data sample
    error =np.array([phi(hs(X[i],w)) - Y[i] for i in range(len(X))])

    if reg == "Ridge" : 
        return (2/n) * np.dot(X.T,error) + lamda * 2 * w
    elif reg == "Sparse" :
        return (2/n) * np.dot(X.T,error) + lamda * np.sign(w) # using sub-gradient
    elif reg == "Elastic" :
        return (1/n) * np.dot(X.T,error) + lamda * (1 - alpha) * w + lamda * alpha * np.sign(w) 
    return (1/n) * np.dot(X.T,(error))

# logistic regression using gradient descent and regularization
def LogisticRegression(X, Y, reg = None, lamda = 0.5, alpha = 0.5, lr = 0.01):
    '''
    description: logistic regression using gradient descent and regularization
    args:
        X: matrix of vectors x_i
        Y: vector of labels (scalars, 1 or 0) y_i
        reg: type of regularization, either Ridge, Sparse, Elastic Net, or None
        lamda: parameter of regularization
        alpha: parameter of Elastic Net regularization
        lr: learning rate of the gradient descent
    return: weight vector w and loss function
    '''
    w = np.zeros(X.shape[1]) # initialize weights vector

    # Lasso and Elastic Net cases needs an iterative search using gradient descent
    Tmax = 200 # upper bound on number of iterations
    for i in range(Tmax):
        dw = gradientReg(X, Y, w, reg=reg, lamda=lamda, alpha=alpha) # compute gradient
        w -= lr * dw # update weights
        sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((i+1)/Tmax))+"-" *int(50 * (1 - (i+1)/Tmax)),int(100*(i+1)/Tmax)))
        sys.stdout.flush() # progress bar
    sys.stdout.write(colorama.Fore.WHITE) # reset color of terminal

    return w, loss(X,Y,w, reg=reg, lamda=lamda,alpha=alpha)