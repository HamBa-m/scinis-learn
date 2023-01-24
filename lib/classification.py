import numpy as np
import sys
import colorama # for colored output in terminal
from .activation import sign
from .loss import loss01, lossAM, cross_entropy, DlossAM, Dcross_entropy
from .transformation import polyMap

################ binary classification ERM algorithms ################
# Single Layer Perceptron
def PLA(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: 
        w: vector of weights after training
        t: number of iterations
    '''
    n, t = len(X), 0
    while loss01(X,Y,w) != 0:
        for i in range(n):
            if sign(X[i], Y[i], w) < 0 : w += X[i]*Y[i]
        t += 1
        print("iter=",t," | loss=",loss01(X,Y,w))
    return w, t

# Single Layer Perceptron with Pocket
def Pocket(X,Y,w):
    '''
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or 0) y_i
        w: vector of weights
    returns: 
        w: vector of weights after training
        t: number of iterations
        loss: average empirical loss
    '''
    n, t = len(X), 0
    Tmax = 500
    w0 = np.array(w)
    while t < Tmax:
        for i in range(n):
            if sign(X[i],Y[i], w0) < 0 : w0 += X[i]*Y[i]
        t += 1
        print("iter=",t," | loss= ",loss01(X,Y,w))
        if loss01(X,Y,w0) < loss01(X,Y,w) : w = w0
    return w, t, loss01(X,Y,w)

# Single Layer Perceptron with Adaline and delta rule
def Adaline(X,Y,w, delta = 0.2):
    '''
    description: Adaline is a Perceptron with a linear activation function
    args:
        X: list of vectors x_i
        Y: list of labels (scalars, 1 or -1) y_i
        w: weights vector
        eps: precision factor
    returns:
        w: vector of weights after training
        t: number of iterations
        loss: loss function value
    '''
    n, t, Tmax = len(X), 0, 100
    lr = 0.0001 # learning rate for the gradient descent
    while abs(DlossAM(X,Y,w)) > delta and t < Tmax :
        print(DlossAM(X,Y,w))
        for i in range(n):
            if (Y[i] - w.T @ X[i]) != 0 : w += 2 * X[i] * (Y[i] - w.T @ X[i]) * lr
        print("iter=",t," | loss=",lossAM(X,Y,w))
        t += 1
    return w, t, lossAM(X,Y,w)

# logistic regression
def LogisticRegression(X, Y, reg = None, lamda = 0.5, alpha = 0.5, lr = 0.1):
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
    Tmax = 2000 # upper bound on number of iterations
    for i in range(Tmax):
        dw = Dcross_entropy(X, Y, w, reg=reg, lamda=lamda, alpha=alpha) # compute gradient
        w -= lr * dw # update weights
        sys.stdout.write(colorama.Fore.GREEN + "\r{1}%\t{0}>".format("="*int(50 * ((i+1)/Tmax))+"-" *int(50 * (1 - (i+1)/Tmax)),int(100*(i+1)/Tmax)))
        sys.stdout.flush() # progress bar
    sys.stdout.write(colorama.Fore.WHITE) # reset color of terminal

    return w, cross_entropy(X,Y,w, reg=reg, lamda=lamda,alpha=alpha)

################ multiclass classification ERM algorithms ################
# One-vs-All
def OVA(data, classes, algo, hyperpara=[]):
    w_list = list()
    for e in classes:
        data_ = np.copy(data)
        row = data_.shape[0]
        for j in range(row):
            if int(data_[j][-1]) != e : data_[j][-1] = -1
            else : data_[j][-1] = 1
        if algo =="SLP":
            # hyperpara = []
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            w, t = PLA(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Pocket":
            # hyperpara = []
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Adaline":
            # hyperpara = [eps]
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            eps = np.copy(hyperpara[0])
            w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
            w_list.append(w)
        elif algo == "Logistic":
            # hyperpara = [lr, Tmax, epsilon]
            col = data_.shape[1]
            lr = np.copy(hyperpara[0])
            Tmax = np.copy(hyperpara[1])
            eps = np.copy(hyperpara[2])
            w, t, ls = LogisticRegression(data_[:,:col-1],data_[:,-1],lr,Tmax,eps)
            w_list.append(w)
        elif algo == "Pocket Trans":
            # hyperpara = [q]
            q = np.copy(hyperpara[0])
            data_ = np.asarray([polyMap(x,q) for x in data_])
            w = np.zeros(data_.shape[1] - 1)
            col = data_.shape[1]
            w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Adaline Trans":
            # hyperpara = [q]
            q = np.copy(hyperpara[0])
            data_ = np.asarray([polyMap(x,q) for x in data_])
            w = np.zeros(data_.shape[1] - 1)
            col = data_.shape[1]
            eps = np.copy(hyperpara[0])
            w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
            w_list.append(w)
    return w_list

# One-vs-One
def OVO(data, classes, algo, hyperpara=[]):
    w_list = list()
    col = data.shape[1]
    for e in classes:
        print(classes,e)
        classes.remove(e)
        for f in classes:
            data_ = []
            for k in range(data.shape[0]):
                if data[k][-1] == e : 
                    l = list(data[k,:col - 1])
                    l.append(1)
                    data_.append(l)
                elif data[k][-1] == f : 
                    l = list(data[k,:col - 1])
                    l.append(-1)
                    data_.append(l)
            data_ = np.asarray(data_)
            if algo =="SLP":
                # hyperpara = []
                w = np.zeros(data.shape[1] - 1)
                col = data_.shape[1]
                w, t = PLA(data_[:,:col-1],data_[:,-1],w)
                w_list.append(w)
            elif algo == "Pocket":
                # hyperpara = []
                w = np.zeros(data.shape[1] - 1)
                col = data_.shape[1]
                w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
                w_list.append(w)
            elif algo == "Adaline":
                # hyperpara = [eps]
                w = np.zeros(data.shape[1] - 1)
                col = data_.shape[1]
                eps = np.copy(hyperpara[0])
                w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
                w_list.append(w)
            elif algo == "Logistic":
                # hyperpara = [lr, Tmax, epsilon]
                col = data_.shape[1]
                lr = np.copy(hyperpara[0])
                Tmax = np.copy(hyperpara[1])
                eps = np.copy(hyperpara[2])
                w, t, ls = LogisticRegression(data_[:,:col-1],data_[:,-1],lr,Tmax,eps)
                w_list.append(w)
            elif algo == "Pocket Trans":
                # hyperpara = [q]
                q = np.copy(hyperpara[0])
                data_ = np.asarray([polyMap(x,q) for x in data_])
                w = np.zeros(data_.shape[1] - 1)
                col = data_.shape[1]
                w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
                w_list.append(w)
            elif algo == "Adaline Trans":
                # hyperpara = [q]
                q = np.copy(hyperpara[0])
                data_ = np.asarray([polyMap(x,q) for x in data_])
                w = np.zeros(data_.shape[1] - 1)
                col = data_.shape[1]
                eps = np.copy(hyperpara[0])
                w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
                w_list.append(w)
    return w_list