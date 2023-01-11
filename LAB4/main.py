import csv
import numpy as np
import matplotlib.pyplot as plt
from nonLinearTransformer import *
from splitData import *

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
        X.append([float(row["x1"]), float(row["x2"]), float(row["y"])])

# split data into training and test sets
train, test = split(X, p = 0.2)

# separate features and labels
train_x, train_y, test_x, test_y = train[:,:2], train[:,-1], test[:,:2], test[:,-1]

# list to keep track of training and test errors
hist = []

for degree in range(1,7):
    train_x_ = np.asarray([psy(x,degree) for x in train_x])
    test_x_ = np.asarray([psy(x,degree) for x in test_x])
    print("Logistic regression starts...")
    print("Degree ", degree, ":")
    print("Training set features: ", train_x_.shape[1])
    w0, t, ls = LogisticRegression(train_x_, train_y, eps = 0.1)
    hist.append((w0, t, "{0:.6f}".format(ls), "{0:.6f}".format(Ls(test_x_, test_y, w0))))
    print("#"*20)
    # plot decision boundary
    plotNonLinearDecisionBoundary2D(train_x, train_y, w0, degree)

# plot training and test error evolution with degree of polynomial transformation
plt.plot(range(1,7), [float(h[2]) for h in hist], label = "training error")
plt.plot(range(1,7), [float(h[3]) for h in hist], label = "test error")
plt.xlabel("degree")
plt.ylabel("loss")
plt.title("training and test error evolution with degree of polynomial transformation")
plt.legend()
plt.show()


