import numpy as np

################### non linear transformation ###################
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
    w = [e for e in x] # list of the form [1, x_1, x_2, ..., x_d]
    for i in range(q-1): 
        for j in range(d):
            for k in range(d):
                w.append(PSI_Q[i][j] * x[k]) 
    return np.array(w)

def polyMap(X, q):
    """
    description: polynomial mapping for multidimensional input
    args:
        X: input matrix
        q: polynom's degree
    return: matrix of the form [1, x_1, x_2, ..., x_d, x_1*x_1, x_1*x_2, ..., x_1*x_d, ..., x_d*x_d, x_1*x_1*x_1, ..., x_1*x_1*x_d, ..., x_d*x_d*x_d, ...]
    """
    return np.array([psy(x,q) for x in X])