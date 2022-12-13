from polynomial_regression import *
import csv

X, Y = [], []
with open('pressure.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array(float(row["temperature"])))
        Y.append(np.array(float(row["pressure"])))

X, Y = np.asarray(X), np.asarray(Y)

# polynomial regression
w0, ls = PolynmialRegression(X, Y, 5)
print(w0, ls)

import matplotlib.pyplot as plt

plt.plot(X[:],Y,"or")

Z = [i for i in range(350)]
# hyperplan equation (in 2D)
def h(wop,x):
    return sum([wop[i] * (x**i) for i in range(len(wop))])

hyper = [h(w0, e) for e in Z]
plt.plot(Z,hyper,"-b" )

plt.show()

