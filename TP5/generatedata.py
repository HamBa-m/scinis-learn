from random import *
import numpy as np
import math
# function that generates uniformly randomized data in 2D
def generateData(data_size,c1,c2,c3,r):
    """
    data_size: number of vectors to generate
    c1, c2, c3: centers coordinates of the 3 data zones (tuples)
    r: radius * 100 of the 3 data zones
    """
    S = []
    for i in range(data_size):
        c = [0,1,2]
        id = choice(c)
        if id == 0:
            x = c1[0] + randrange(-r,r)/20
            y = math.sqrt((r/20)**2 - x**2)
            S.append(np.array([1,x,y, 1]))
        elif id == 1:
            x = c2[0] + randrange(-r,r)/20
            y = math.sqrt((r/20)**2 - x**2)
            S.append(np.array([1,x,y, 2]))
        else:
            x = c3[0] + randrange(-r,r)/20
            y = math.sqrt((r/20)**2 - x**2)
            S.append(np.array([1,x,y, 3]))
    return np.asarray(S)

X =  generateData(500, (5,5), (16,16), (16,5), 100)
print(X)

np.savetxt("data_sep_2D.csv", X, delimiter=",",fmt="%d")