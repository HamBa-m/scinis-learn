import csv
import numpy as np

X, Y = [], []
with open('data_sep_2D.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array([int(row["x0"]),int(row["x1"]),int(row["x2"])]))
        Y.append(np.array([int(row["y"])]))

import matplotlib.pyplot as plt

for i in range(len(X)):
    if Y[i] == 1 :
        plt.plot(X[i][1],X[i][2], "or")
    elif Y[i] == 2 :
        plt.plot(X[i][1],X[i][2], "og")
    else :
        plt.plot(X[i][1],X[i][2], "ob")

plt.show()