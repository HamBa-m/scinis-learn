from logistic_regression import *
import csv

X, Y = [], []
with open('binary.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(np.array([1,float(row["gre"]),float(row["gpa"])]))
        Y.append(np.array([int(row["admit"])]))

# tranformation functions to normalize data

# Z-value transformer
def Zvalue(x,mean,std):
    return (x - mean)/std

# data normalization
X, Y = np.asarray(X), np.asarray(Y)
meanX, meanY, stdX, stdY = np.mean(X[:,1]), np.mean(X[:,2]), np.std(X[:,1]), np.std(X[:,2])

for i in range(len(X)):
    # using z-value tranformer
    X[i][1], X[i][2] = Zvalue(X[i][1], meanX, stdX), Zvalue(X[i][2], meanY, stdY)


w0, t, ls = LogisticRegression(X, Y)
print("FINAL RESULTS:")
print("optimal weight vector: ", w0,"\t| iterations: ", t, "\t| empirical loss: ", "{0:.6f}".format(ls))

import matplotlib.pyplot as plt
from matplotlib import cm
# hyperplan equation (in 3D)
def hyperplan3D(wop,x,y):
    return 1/( 1 + np.exp(-(wop[1] * x + wop[2] * y + wop[0])))


# Create the figure
fig = plt.figure()
# Add an axes
ax = fig.add_subplot(111,projection='3d')
for e in range(len(X)):
    if Y[e] == 1 :ax.plot(X[e][1], X[e][2], 1, "o", color = 'red')
    else : ax.plot(X[e][1], X[e][2], 0, "o", color = 'green')

x, y = np.arange(-4,2,0.1),np.arange(-4,2,0.1)
x, y = np.meshgrid(x,y)
z = hyperplan3D(w0, x, y)
surf = ax.plot_surface(x, y, z,cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha= 0.6)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_xlabel("GRE")
ax.set_ylabel("GPA")
ax.set_zlabel("probability")
ax.set_title("Plot of Decision Boundary with Logistic Regression.\nEmpirical Loss: "+"{0:.6f}".format(ls)+" | nbr of iterations: "+str(t))

plt.show()

