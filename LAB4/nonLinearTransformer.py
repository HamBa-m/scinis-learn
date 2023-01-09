import numpy as np

# sigmoid activation function
def hw(x,w):
    '''
    description: sigmoid activation function
    args:
        x: a vector
        w: weight vector
    return: sigmoid of x
    '''
    return 1/(1 + np.exp(-w.T @ x))

# empirical loss of the logistic regression
def Ls(X,Y,w):
    """
    description: empirical loss of the logistic regression
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
    return: empirical loss of the logistic regression
    """
    return np.mean( [-Y[i] * np.log(hw(X[i], w)) - (1 - Y[i]) * np.log(1 - hw(X[i], w)) for i in range(len(Y))] )

# gradient of cost function
def DLs(X,Y,w):
    """
    description: gradient of cost function
    args:
        x: a vector x_i
        y: the label y_i associated
        w: weight vector
    return: gradient of cost function
    """
    grad = [] # gradient vector
    d, m = len(w), len(Y) # dimension of data, size of data sample
    for j in range(d): 
        grad.append( np.mean([ (hw(X[i], w) - Y[i])*X[i][j] for i in range(m) ]) )
    return np.array(grad)

# logistic regression algorithm
def LogisticRegression(X, Y, lr = 0.1, Tmax = 1000, eps = 0.2):
    """
    description: logistic regression algorithm
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        lr: learning rate
        Tmax: maximum number of iterations
        eps: threshold for stopping criterion (precision factor)
    return:
        w: vector of weights after training
        t: number of iterations
        ls: empirical loss
    """
    t = 0 # iteration counter
    w = np.zeros(X.shape[1]) # initialize weights vector
    ls = Ls(X,Y,w) # empirical loss
    while(ls > eps and t < Tmax): # stopping criterion
        if not(t%100) : print("iter:",t,"\t| empirical loss: ", "{0:.6f}".format(ls)) # print loss every 100 iterations
        w -= lr * DLs(X,Y,w) # update weights with gradient descent
        ls = Ls(X,Y,w) # update empirical loss
        t += 1 # increment iteration counter
    return w, t, ls

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

# define a function to plot the decision boundary (NOT FINAL)
def plotDecisionBoundary(w0, X_, axes):
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    U,V = np.meshgrid(u,v)
    # convert U, V to vectors for calculating additional features
    # using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))
    
    Z = X_.dot(w0)
    
    # reshape U, V, Z back to matrix
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))
    
    cs = axes.contour(U,V,Z,levels=[0],cmap= "Greys_r")
    axes.legend(labels=['1', '0', 'Decision Boundary'])
    return cs