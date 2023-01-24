import numpy as np
from random import choice, uniform
import math
import csv

################ Data Generation ################
def generateData(data_size, circles, dim = 2, filename="data.csv"):
    """
    data_size: number of vectors to generate
    circles: a list of dictionaries, each containing the center coordinates, class label, and radius for a circle
    filename: name of the file to save the data to
    """
    S = []
    for i in range(data_size):
        circle = choice(circles)
        x = circle['center'][0] + uniform(-circle['radius']*10, circle['radius']*10)/10
        y = circle['center'][1] + uniform(-circle['radius']*10, circle['radius']*10)/10
        if dim == 3:
            z = circle['center'][2] + uniform(-circle['radius']*10, circle['radius']*10)/10
            if math.sqrt((x - circle['center'][0])**2 + (y - circle['center'][1])**2 + (z - circle['center'][2])**2) > circle['radius']:
                # If the point is outside the circle, try again
                continue
            S.append(np.array([1, x, y, z, circle['class']]))
        elif dim == 2:
            if math.sqrt((x - circle['center'][0])**2 + (y - circle['center'][1])**2) > circle['radius']:
                # If the point is outside the circle, try again
                continue
            S.append(np.array([1, x, y, circle['class']]))
        else: 
            print("Error: dim must be 2 or 3")
            return None 
    np.savetxt(filename, S, delimiter=",",fmt="%f")
    return np.asarray(S)

def makeRegressionData(data_size, filename="data.csv", degree = 1):
    """
    description: generates data for a regression problem with a polynomial function
    args:
        data_size: number of vectors to generate
        filename: name of the file to save the data to
        degree: degree of the polynomial function
    return: a numpy array containing the data
    """
    S = []
    for i in range(data_size):
        x = uniform(-1, 1)
        y = 2*(x**degree) + uniform(-0.3, 0.3)
        S.append(np.array([1, x, y]))
    np.savetxt(filename, S, delimiter=",",fmt="%f")
    return np.asarray(S)

######################### Data loading #########################
def loadData(filename, features):
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