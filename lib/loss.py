import numpy as np
from .hypothesis import h
from .activation import sigmoid, sign

################ empirical loss functions ################
# 0-1 loss
def loss01(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: average empirical loss
    '''
    n = len(X) # size of data sample
    misclassified = [1 if sign(X[i], Y[i], w) != 1  else 0 for i in range(len(X))]
    return sum(misclassified)/n

# Arithmetical Mean (AM)
def lossAM(X,Y,w, reg=None, lamda=0.5, alpha = 0.5):
    """
    description: AM loss function with regularization variants
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
    error = [(Y[i] - w.T @ X[i])**2 for i in range(len(X))] # squared error vector
    if reg == "Ridge" : 
        return np.sum(error)/n + lamda * np.sum(np.power(w,2)) 
    elif reg == "Lasso" :
        return np.sum(error)/n + lamda * np.sum(np.array([abs(e) for e in w])) 
    elif reg == "Elastic" :
        return np.sum(error)/(2*n) + lamda * (1 - alpha)/2 * np.sum(np.power(w,2)) + lamda * alpha * np.sum(np.array([abs(e) for e in w]))
    
    return np.sum(error)/n


# cross-entropy 
def cross_entropy(X,Y,w, reg=None, lamda=0.5, alpha = 0.5):
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
    error = [- Y[i] * np.log(sigmoid(h(X[i],w))) for i in range(len(X))]
    if reg == "Elastic":
        return np.sum(error)/n + (lamda * alpha) * np.sum(np.array([abs(e) for e in w]))+ (lamda * (1 - alpha) /(2 *n)) * np.sum(np.power(w,2))
    elif reg == "Ridge" : 
        return np.sum(error)/n + (lamda/(2 *n)) * np.sum(np.array([abs(e) for e in w]))
    elif reg == "Sparse" :
        return np.sum(error)/n + (lamda/(2 *n)) * np.sum(np.power(w,2))

    return np.sum(error)/n

################ gradients ################
# AM loss function
def DlossAM(X,Y,w, reg=None, lamda = 0.5, alpha = 0.5):
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
    error = [(Y[i] - w.T @ X[i]) for i in range(len(X))] # error vector 
    if reg == "Ridge" : 
        return (2/n) * np.dot(X.T,error) + lamda * 2 * w # using gradient
    elif reg == "Lasso" :
        return (2/n) * np.dot(X.T,error) + lamda * np.sign(w) # using sub-gradient
    elif reg == "Elastic" :
        return (1/n) * np.dot(X.T,error) + lamda * (1 - alpha) * w + lamda * alpha * np.sign(w)

    return (2/n) * np.dot(X.T,error)

# cross-entropy loss function
def Dcross_entropy(X,Y,w, reg=None, lamda = 0.5, alpha = 0.5):
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
    error = np.array([sigmoid(h(X[i],w)) - Y[i] for i in range(len(X))])

    if reg == "Ridge" : 
        return (2/n) * np.dot(X.T,error) + lamda * 2 * w
    elif reg == "Sparse" :
        return (2/n) * np.dot(X.T,error) + lamda * np.sign(w) # using sub-gradient
    elif reg == "Elastic" :
        return (1/n) * np.dot(X.T,error) + lamda * (1 - alpha) * w + lamda * alpha * np.sign(w)

    return (1/n) * np.dot(X.T,error)