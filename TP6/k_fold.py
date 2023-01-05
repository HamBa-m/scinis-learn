import csv
import numpy as np
from math import sqrt
from random import randrange


def divisors(m):
    div = set([m])
    for i in range(2, int(sqrt(m))+1):
        if m % i == 0:
            div.add(i)
            div.add(m//i)
    return list(div)


def k_fold(data):
    partitions = list()
    data_ = list(data)
    m = len(data)
    div = divisors(m)
    print("please choose a k value among the following:")
    for e in div:
        print(e, end=" ")
    print()
    k = int(input())
    for i in range(k):
        l = list()
        while len(l) < int(m/k):
            ind = randrange(len(data_))
            l.append(data_.pop(ind))
        partitions.append(l)
    return partitions