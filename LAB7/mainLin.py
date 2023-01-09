from regularization import *
from validation import *
import numpy as np
import csv

# data extraction
X, Y = [], []
col = "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income".split(",")
with open('California_House_Price.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        cdt = False
        for e in row.values():
            if e == "": 
                cdt = True
                break
        if cdt : continue
        new = [1]+[float(row[col[i]]) for i in range(len(col))]
        if row["ocean_proximity"] == "INLAND" : new += [3]
        elif row["ocean_proximity"] == "NEAR BAY" : new += [1]
        else : new += [2]
        X.append(np.array(new + [float(row["median_house_value"])]))

# data normalization
# Z-value transformer
def Zvalue(x,mean,std):
    return (x - mean)/std

def dataNormalize(data):
    X = np.asarray(data)
    mean, std = [], []
    for i in range(X.shape[1]):
        mean.append(np.mean(X[:,i]))
        std.append(np.std(X[:,i]))
    for i in range(X.shape[0]):
        for j in range(1,X.shape[1]):
            X[i][j] = Zvalue(X[i][j],mean[j],std[j])
    return X

# data normalization
X = dataNormalize(X)

# data spliting 1:5
train, test = split(X)
X, Y = train[:,:10], train[:,10]
X_, Y_ = test[:,:10], test[:,10]

# Simple Linear Regression
w0, ls = LinearRegression(X, Y)
ld = loss(X_, Y_, w0)
print("FINAL RESULTS:")
print("Simple Linear Regression:")
print("empirical loss: ", "{0:.6f}".format(ls))
print("generalization loss: ", "{0:.6f}".format(ld))
print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))

# Ridge Regression
print("Ridge Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Ridge", lamda= lamda)
    ld = loss(X_, Y_, w0, reg="Ridge", lamda=lamda)
    print("empirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))

# Lasso Regression
print("Lasso Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Lasso", lamda= lamda, lr=0.01)
    ld = loss(X_, Y_, w0, reg="Lasso", lamda=lamda)
    print("\nempirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))

# Elastic Net Regression
print("Elastic Net Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Lasso", lamda= lamda, lr=0.01)
    ld = loss(X_, Y_, w0, reg="Lasso", lamda=lamda)
    print("\nempirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))