import numpy as np
import numpy.linalg as lg

# polynomial regression mapping for 1 dimension
def phi(x, n):
    """
    description: polynomial regression mapping for 1 dimension
    args:
        x: the entry variable
        n: the degree of the output polynome
    return: the vector of the form [1, x, x^2, ..., x^n]
    """
    return np.array([x**i for i in range(n+1)]).T

# polynomial regression hypothesis 
def hs(x,w):
    """
    description: polynomial regression hypothesis
    args:
        x: vector of features
        w: vector of weights
    return: matrcial product of x and w.T
    """
    return w.T @ x

# error function between real y and predicted y
def e(x,y,w):
    """
    description: error function between real y and predicted y
    args:
        x: vector of features
        y: label (scalar, 1 or -1)
        w: vector of weights
    return: error between real y and predicted y
    """
    return y - hs(x,w)

# empirical error function
def loss(X,Y,w):
    '''
    description: empirical error function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: vector of weights
    return: average empirical error
    '''
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))] # Mean-Squared Error (MSE)
    return np.sum(error)/n

# Linear regression for polynomial tasks (1 dimension)
def LinearRegressionforPoly(X,Y,n):
    '''
    description: Linear regression for polynomial tasks (1 dimension)
    args:
        X: vector of scalars x_i
        Y: vector of scalars y_i (labels, 1 or 0)
        n: degree of polynomial mapping
    return: vector of weights w and empirical error
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

# polynomial mapping for multidimensional input
def psy(x,q):
    """
    description: polynomial mapping for multidimensional input
    args:
        x: input vector
        q: polynom's degree
    return: vector of the form [1, x_1, x_2, ..., x_d, x_1*x_1, x_1*x_2, ..., x_1*x_d, ..., x_d*x_d, x_1*x_1*x_1, ..., x_1*x_1*x_d, ..., x_d*x_d*x_d, ...]
    """
    PSI_Q = [np.power(x,i) for i in range(1,q)] # list of all possible combinations of x_i
    d = x.size # dimension of input vector
    w = [1] + [e for e in x] # list of the form [1, x_1, x_2, ..., x_d]
    for i in range(q-1): 
        for j in range(d):
            for k in range(d):
                w.append(PSI_Q[i][j] * x[k]) 
    return np.array(w)

# polynomial regression 
def PolynomialRegression(X,Y,q):
    """
    description: polynomial regression for multidimensional input using algebraic solution
    args:
        X: list of vectors x_i
        Y: list of label y_i
        n: polynom's degree
    return: vector of weights w and empirical error
    """
    # mapping the dataset to an upper degree
    X_ = np.asarray([psy(x,q) for x in X])
    # computes the Hessian matrix of the loss function applied to the hypothesis
    A = np.dot(X_.T,X_) 
    # computes the second term of the linear system A.w = b
    b = np.dot(X_.T, Y) 
    # computes the pseudoinverse of A using a Singular-Value Decomposition algorithm
    Aplus = np.linalg.pinv(A)
    # solve the linear system A+.w = b
    w = np.dot(Aplus,b)
    return w, loss(X_,Y,w)