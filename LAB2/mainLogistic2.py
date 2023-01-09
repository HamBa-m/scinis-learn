from logistic_regression import *
import csv

X, Y = [], []
with open('binary.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array([1,float(row["gre"]),float(row["gpa"]), float(row["rank"])]))
        Y.append(np.array([int(row["admit"])]))

# tranformation functions to normalize data
def trMinMax(x, m, M):
    return (x - m)/(M - m)

# data normalization
X, Y = np.asarray(X), np.asarray(Y)
mx, Mx, my, My, mz, Mz = np.min(X[:,1]),np.max(X[:,1]),np.min(X[:,2]),np.max(X[:,2]),np.min(X[:,3]),np.max(X[:,3])

for i in range(len(X)):
    # using max-min tranformer
    X[i][1], X[i][2], X[i][3] = trMinMax(X[i][1],mx,Mx), trMinMax(X[i][2], my,My), trMinMax(X[i][3], mz,Mz)

w0, t, ls = LogisticRegression(X, Y, Tmax = 1000)
print("FINAL RESULTS:")
print("optimal weight vector: ", w0,"\t| iterations: ", t, "\t| empirical loss: ", "{0:.6f}".format(ls))