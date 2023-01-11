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
    training = np.copy(data) # copy of data to avoid modifying data
    validation = list() 
    m = len(data) 
    k = int(m * p) # number of data points in validation set
    for i in range(k):
        ind = randrange(len(training)) # choose a random index in training set 
        validation.append(training[ind]) # add the data point at the index to the validation set 
        training = np.delete(training, ind, 0) # remove the data point at the index from the training set
    return np.asarray(training), np.asarray(validation)
