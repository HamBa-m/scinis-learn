import numpy as np
from random import randrange

def divisors(m):
    """
    description: list of divisors of m
    args:
        m: integer
    return: list of divisors of m
    """
    div = set([m]) # set of divisors
    for i in range(2, int(m**(0.5))+1): # divisors are between 2 and sqrt(m)
        if m % i == 0: # if i is a divisor of m
            div.add(i) # add i to the set
            div.add(m//i) # add m//i to the set
    return list(div)

def k_folds(data):
    """
    description: k-fold cross validation partitions of data
    args:
        data: list of data points
    return: list of k partitions of data
    """
    partitions = list() # list of k partitions of data
    data_ = list(data) # copy of data to avoid modifying data
    m = len(data) # size of data
    div = divisors(m) # list of divisors of m (k values)
    print("please choose a k value among the following:") # ask user to choose a k value
    for e in div: # print divisors of m (allowed k values)
        print(e, end=" ")
    print()
    k = int(input()) # read k value from user 
    for i in range(k): # for each partition 
        l = list() # list of data points in the partition
        while len(l) < int(m/k): # while the partition is not full
            ind = randrange(len(data_)) # choose a random index
            l.append(data_.pop(ind)) # add the data point at the index to the partition
        partitions.append(l) # add the partition to the list of partitions
    return np.asarray(partitions)

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