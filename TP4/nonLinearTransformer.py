import numpy as np

# sigmoid activation function
def hw(x,w):
    '''
    x: a vector
    '''
    return 1/(1 + np.exp(-w.T @ x))

# cost function of the logistic regression
def Ls(X,Y,w):
    """
    X: a list of vectors x_i
    Y: the list of labels y_i associated
    w: weight vector
    """
    return np.mean( [-Y[i] * np.log(hw(X[i], w)) - (1 - Y[i]) * np.log(1 - hw(X[i], w)) for i in range(len(Y))] )

# gradient of cost function
def DLs(X,Y,w):
    """
    x: a vector x_i
    y: the label y_i associated
    w: weight vector
    """
    grad = []
    d, m = len(w), len(Y)
    for j in range(d):
        grad.append( np.mean([ (hw(X[i], w) - Y[i])*X[i][j] for i in range(m) ]) )
    return np.array(grad)

# logistic regression algorithm
def LogisticRegression(X, Y, lr = 0.1, Tmax = 1000, eps = 0.2):
    t = 0 #initialisation de compteur
    w = np.zeros(X.shape[1])
    ls = Ls(X,Y,w)
    while(ls > eps and t < Tmax):
        if not(t%100) : print("iter:",t,"\t| empirical loss: ", "{0:.6f}".format(ls))
        w -= lr * DLs(X,Y,w)
        ls = Ls(X,Y,w)
        t += 1
    return w, t, ls

# polynomial mapping for multidimensional input
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