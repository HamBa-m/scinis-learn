from regularization import *
from validation import *
import numpy as np
import csv

# data extraction
X, Y = [], []
col = "longitude,latitude,housing_median_age,total_rooms,total_bedrooms,population,households,median_income".split(",")
with open('Tatanic.csv', mode='r') as csv_file:
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

# Ridge logistic Regression
print("Ridge Logistic Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Ridge", lamda= lamda, lr=0.01)
    ld = loss(X_, Y_, w0, reg="Ridge", lamda=lamda)
    print("\nempirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))

# Sparse logistic Regression
print("Sparse Logistic Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Sparse", lamda= lamda, lr=0.01)
    ld = loss(X_, Y_, w0, reg="Sparse", lamda=lamda)
    print("\nempirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))

# Elatsic Net logistic Regression
print("Elatsic Net Logistic Regression:")
for lamda in [0.01,0.1,0.2,0.5,1,2]:
    w0, ls = LinearRegression(X, Y, reg="Elastic", lamda= lamda, lr=0.01)
    ld = loss(X_, Y_, w0, reg="Elastic", lamda=lamda)
    print("\nempirical loss: ", "{0:.6f}".format(ls),"\t | lamda ", lamda)
    print("generalization loss: ", "{0:.6f}".format(ld))
    print("|Ld - Ls| = ", "{0:.6f}".format(abs(ld - ls)))