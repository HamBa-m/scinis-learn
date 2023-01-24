from polynomial_regression import *
import csv

X1, X2, Y = [], [], []
with open('pressure.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X1.append(np.array(float(row["temperature"])))
        X2.append(np.array([float(row["temperature"])]))
        Y.append(np.array(float(row["pressure"])))

X1, X2, Y = np.asarray(X1), np.asarray(X2), np.asarray(Y)

# linear regression for polynomial tasks
w0, ls = LinearRegressionforPoly(X1, Y, 4)
print("FINAL RESULTS 1:")
print("optimal weight vector: ", w0, "\t| empirical loss: ", "{0:.6f}".format(ls))

# polynomial regression
w0, ls = PolynomialRegression(X2, Y, 5)
print("FINAL RESULTS 2:")
print("optimal weight vector: ", w0, "\t| empirical loss: ", "{0:.6f}".format(ls))

import matplotlib.pyplot as plt

plt.plot(X1[:],Y,"or")

Z = [i for i in range(370)]
# hyperplan equation (in 2D)
def h(wop,x):
    return sum([wop[i] * (x**i) for i in range(len(wop))])

hyper = [h(w0, e) for e in Z]
plt.plot(Z,hyper,"-b" )
plt.xlabel("temperature")
plt.ylabel("pressure")
plt.title("polynomial regression hypothesis\ndegree = "+str(5)+", empirical loss = "+"{0:.6f}".format(ls))
plt.show()