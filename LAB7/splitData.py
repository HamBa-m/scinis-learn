import numpy as np
from random import randrange

def split(data, p = 0.2):
    """
    description: split data into training and test sets
    args:
        data: list of data points
        p: proportion of data in test set
    return: training and test sets
    """
    training = list(data) # copy of data to avoid modifying data
    test = list() 
    m = len(data) 
    k = int(m * p) # number of data points in test set
    for i in range(k):
        ind = randrange(len(training)) # choose a random index in training set 
        test.append(training.pop(ind)) # add the data point at the index to the test set 
    return training, test