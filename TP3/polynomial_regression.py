import numpy as np
import numpy.linalg as lg

# polynomial regression mapping for 1 dimension
def phi(x, n):
    """
    x: the entry variable
    n: the degree of the output polynome
    """
    return np.array([x**i for i in range(n+1)]).T

# polynomial regression hypothesis 
def hs(x,w):
    return w.T @ x

# cost function
def e(x,y,w):
    return y - w.T @ x

# empirical error function
def loss(X,Y,w):
    n = len(X) # size of data sample
    error = [(e(X[i], Y[i], w))**2 for i in range(len(X))]
    return np.sum(error)/n

# Linear regression for polynomial tasks (1 dimension)
def LinearRegressionforPoly(X,Y,n):
    '''
    X: vector of scalars x_i
    Y: vector of scalars y_i (labels, 1 or 0)
    n: degree of polynomial mapping
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

# polynomial regression mapping for multidimensional input
def psy(x,q):
    """
    x: input vector
    q: polynom's degree
    """
    PSI_Q = [np.power(x,i) for i in range(1,q)]
    d = x.size
    w = [1] + [e for e in x]
    for i in range(q-1):
        for j in range(d):
            for k in range(d):
                w.append(PSI_Q[i][j] * x[k])
    return np.array(w)

# polynomial regression 
def PolynomialRegression(X,Y,q):
    """
    X: list of vectors x_i
    Y: list of label y_i
    n: polynom's degree
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
