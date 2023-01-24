import numpy as np

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