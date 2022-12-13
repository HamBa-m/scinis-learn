from linear_regression import *
import csv

X, Y = [], []
csv_reader = csv.reader(open("pop.csv"), delimiter=";")
for row in csv_reader:
    if row[0] == "X1" : continue
    if row[3] == "" : continue
    X.append(np.array([1,float(row[0].replace(',', '.')),float(row[1].replace(',', '.')),float(row[2].replace(',', '.')),float(row[3].replace(',', '.'))]))
    Y.append(np.array([float(row[4])]))

X, Y = np.asarray(X), np.asarray(Y)

w0, ls = LinearRegression(X, Y)
print("FINAL RESULTS:")
print("optimal weight vector: ", w0.T,"\t| empirical loss: ", "{0:.6f}".format(ls))