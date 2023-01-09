import numpy as np
from random import randrange

def split(data, p = 0.2):
    """
    description: split data into training and validation sets
    args:
        data: list of data points
        p: proportion of data in validation set
    return: training and validation sets
    """
    training = list(data) # copy of data to avoid modifying data
    validation = list() 
    m = len(data) 
    k = int(m * p) # number of data points in validation set
    for i in range(k):
        ind = randrange(len(training)) # choose a random index in training set 
        validation.append(training.pop(ind)) # add the data point at the index to the validation set 
    return training, validation