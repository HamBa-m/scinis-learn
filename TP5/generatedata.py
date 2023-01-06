import numpy as np
from random import choice, randrange, uniform
import math

def generate_data(data_size, circles, r):
    """
    data_size: number of vectors to generate
    circles: a list of dictionaries, each containing the center coordinates and class label for a circle
    r: radius of the circles
    """
    S = []
    for i in range(data_size):
        circle = choice(circles)
        x = circle['center'][0] + uniform(-r*10, r*10)/10
        y = circle['center'][1] + uniform(-r*10, r*10)/10
        if math.sqrt((x - circle['center'][0])**2 + (y - circle['center'][1])**2) > r:
            # If the point is outside the circle, try again
            continue
        S.append(np.array([1, x, y, circle['class']]))
    return np.asarray(S)

# Define the center coordinates and class labels for each circle
circles = [
    {'center': (0, 0), 'class': 1},
    {'center': (8, 2.5), 'class': 2},
    {'center': (0, 10), 'class': 3}
]

# Generate a dataset with 1000 samples
data_size = 500
r = 2.9
data = generate_data(data_size, circles, r)

import matplotlib.pyplot as plt

# Extract the x and y coordinates and class labels from the data
x = data[:, 1]
y = data[:, 2]
labels = data[:, 3]

# Set the colors for each class
colors = ['red', 'green', 'blue']

# Create a scatter plot
plt.scatter(x, y, c=[colors[int(label)-1] for label in labels])

# Show the plot
plt.show()

np.savetxt("data_sep_2D.csv", data, delimiter=",",fmt="%f")