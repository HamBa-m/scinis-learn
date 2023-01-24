from perceptron import *
from pocket import *
from adaline import *
import csv

X, Y = [], []
with open('data_sep_2D.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array([int(row["x0"]),int(row["x1"]),int(row["x2"])]))
        Y.append(np.array([int(row["y"])]))

w = np.zeros(3)
# w0, t, ls = Adaline(X, Y, w, eps = 0.3)
w0, t = PLA(X, Y, w)
# w0, t, ls = Pocket(X, Y, w)
print(w0, t)

import matplotlib.pyplot as plt

for i in range(len(X)):
    if Y[i] == 1 :
        plt.plot(X[i][1],X[i][2], "or")
    else :
        plt.plot(X[i][1],X[i][2], "og")

Z = [i for i in range(21)]
# hyperplan equation (in 2D)
def h(wop,x):
    return -(wop[1]*x + wop[0])/wop[2]

hyper = [h(w0, e) for e in Z]
plt.plot(Z,hyper,"-b" )

plt.show()

