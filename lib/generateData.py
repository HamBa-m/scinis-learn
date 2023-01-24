import numpy as np
from random import choice, uniform
import math

def generate_data(data_size, circles, dim = 2, filename="data.csv"):
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