import numpy as np
from random import choice, uniform
import math
import csv
from typing import List, Dict, Any, Optional


################ Data Generation ################

def generateData(data_size: int, circles: List[Dict[str, Any]], dim: int = 2, filename: str = "data.csv") -> np.ndarray:
    if dim not in [2, 3]:
        print("Error: dim must be 2 or 3")
        return

    num_cols = dim + 2
    S = np.zeros((data_size, num_cols))
    i = 0

    if dim == 2:        
        while i < data_size:
            circle = choice(circles)
            center = circle['center']
            radius = circle['radius']
            clss = circle['class']
            
            x = center[0] + uniform(-radius, radius)
            y = center[1] + uniform(-radius, radius)
            
            if (x - center[0])**2 + (y - center[1])**2 <= radius**2:
                S[i] = [1, x, y, clss]
                i += 1

    elif dim == 3:        
        while i < data_size:
            circle = choice(circles)
            center = circle['center']
            radius = circle['radius']
            clss = circle['class']
            
            x = center[0] + uniform(-radius, radius)
            y = center[1] + uniform(-radius, radius)
            z = center[2] + uniform(-radius, radius)

            if (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2:
                S[i] = [1, x, y, z, clss]
                i += 1

    np.savetxt(filename, S, delimiter=",",fmt="%f")
    return S

def makeRegressionData(data_size: int, filename: str = "data.csv", degree: int = 1) -> np.ndarray:
    """
    description: generates data for a regression problem with a polynomial function
    args:
        data_size: number of vectors to generate
        filename: name of the file to save the data to
        degree: degree of the polynomial function
    return: a numpy array containing the data
    """
    S = np.zeros((data_size, 3))
    for i in range(data_size):
        x = uniform(-1, 1)
        y = 2*(x**degree) + uniform(-0.3, 0.3)
        S[i] = [1, x, y]
    np.savetxt(filename, S, delimiter=",",fmt="%f")
    return S

######################### Data loading #########################
def loadData(filename: str, features: List[str]) -> np.ndarray:
    """
    args:
        filename: name of the file to load the data from
        features: list of features to load
    return: a numpy array containing the data
    """
    data = []

    # open and format data
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            data.append([float(row[features[i]]) for i in range(len(features))])

    return np.asarray(data)
