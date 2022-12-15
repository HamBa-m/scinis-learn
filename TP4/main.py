import csv
import numpy as np
import matplotlib.pyplot as plt
from nonLinearTransformer import *

X1, X2, Y1, Y2, X, Y = [],[],[],[],[],[]

# open and format data 
with open('ex2data2.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        if row["y"] == "1": 
            X1.append(float(row["x1"]))
            Y1.append(float(row["x2"]))
        else : 
            X2.append(float(row["x1"]))
            Y2.append(float(row["x2"]))
        X.append([1,float(row["x1"]), float(row["y"])])
        Y.append(float(row["y"]))

X, Y = np.asarray(X), np.asarray(Y)

# X_ = np.asarray([psy(x,2) for x in X])
w0, t, ls = LogisticRegression(X, Y, eps = 0.1)
print("FINAL RESULTS:")
print("optimal weight vector: ", w0,"\t| empirical loss: ", "{0:.6f}".format(ls))

# plot results
plt.plot(X1,Y1,'+', color="black", label = 'y = 1')
plt.plot(X2,Y2,'o', color="yellow", label = 'y = 0')


plt.xlabel("Microship Test 1")
plt.ylabel("Microship Test 2")
plt.legend()
plt.title("data plot")
plt.show()