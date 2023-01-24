from random import *
import numpy as np
# function that generates uniformly randomized data in 2D
def generateData(data_size,x1,x2,x3,x4,y1,y2,y3,y4):
    """
    description: generates uniformly randomized data in 2D
    args:
        data_size: number of vectors to generate
        x1: lower bound of interval 1 of data on the x-axis
        x2: upper bound of interval 1 of data on the x-axis
        y1: lower bound of interval 1 of data on the y-axis
        y2: upper bound of interval 1 of data on the y-axis
        x3: lower bound of interval 2 of data on the x-axis
        x4: upper bound of interval 2 of data on the x-axis
        y3: lower bound of interval 2 of data on the y-axis
        y4: upper bound of interval 2 of data on the y-axis
    returns: list of vectors of the form [1, x, y, label] where label is 1 if the point is in the first interval and -1 if it is in the second interval
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
np.savetxt("data_nsep_2D.csv", X, delimiter=",",fmt="%d")