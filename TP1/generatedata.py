from random import *
import numpy as np
# function that generates uniformly randomized data in 2D
def generateData(data_size,x1,x2,x3,x4,y1,y2,y3,y4):
    """
    data_size: number of vectors to generate
    x1: lower bound of interval 1 of data on the x-axis
    x2: upper bound of interval 1 of data on the x-axis
    y1: lower bound of interval 1 of data on the y-axis
    y2: upper bound of interval 1 of data on the y-axis
    """
    S = []
    for i in range(data_size):
        c = [0,1]
        if choice(c):
            xi = [1,randint(x1,x2), randint(y1,y2), 1]
            yi = 1
        else:
            xi = [1,randint(x3,x4), randint(y3,y4), -1]
            yi = -1
        S.append(np.array(xi))
    return np.asarray(S)

X =  generateData(50, 0, 15, 5, 20, 0, 15, 5, 20)
print(X)

np.savetxt("data_nsep_2D.csv", X, delimiter=",",fmt="%d")