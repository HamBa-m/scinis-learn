from linear_regression import *
import csv

X, Y = [], []
with open('car data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array([1,int(row["speed"])]))
        Y.append(np.array([int(row["dist"])]))

X, Y = np.asarray(X), np.asarray(Y)

w0, ls = LinearRegression(X, Y)
print("FINAL RESULTS:")
print("optimal weight vector: ", w0,"\t| empirical loss: ", "{0:.6f}".format(ls))

import matplotlib.pyplot as plt

plt.plot(X[:,1],Y,"or", label="real data")

Z = [i for i in range(4,28)]
# hyperplan equation (in 2D)
def h(wop,x):
    return wop[1] * x + wop[0]

hyper = [h(w0, e) for e in Z]
plt.plot(Z,hyper,"-b", label="regression line")

plt.legend()
plt.title("the real distance and the estimated one with respect to the speed")
plt.show()

