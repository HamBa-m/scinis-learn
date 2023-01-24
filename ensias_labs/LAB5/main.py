import csv
import numpy as np
from algos.nonLinearTransformer import *
from ova import *
from ovo import *

# helper functions
def loss(Y, Y_):
    s = 0
    for i in range(Y.shape[0]):
        if Y[i] != Y_[i] : s += 1
    return s/len(Y)

def predict(data, w_list):
    pred = []
    for e in data:
        pred.append(np.argmax([np.dot(w, e) for w in w_list])+1)
    return pred

data = []
# separable data
with open('data_sep_2D.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        data.append(np.array([float(row["x0"]),float(row["x1"]),float(row["x2"]),float(row["y"])]))

# noisy data
# with open('data_noise_2D.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         data.append(np.array([float(row["x0"]),float(row["x1"]),float(row["x2"]),float(row["y"])]))

# non separable data
# with open('data_nsep_2D.csv', mode='r') as csv_file:
#     csv_reader = csv.DictReader(csv_file)
#     for row in csv_reader:
#         data.append(np.array([float(row["x0"]),float(row["x1"]),float(row["x2"]),float(row["y"])]))


data = np.asarray(data)

# testing One vs All
# w_list = OVA(data,[1,2,3],"SLP")
w_list = OVA(data,[1,2,3],"Pocket")
# w_list = OVA(data,[1,2,3],"Adaline",[0.1])
# w_list = OVA(data,[1,2,3],"Logistic",[0.05, 500, 0.1])
# w_list = OVA(data,[1,2,3],"Pocket Trans",[5])
# w_list = OVA(data,[1,2,3],"Adaline Trans",[5])

# testing One vs One
# w_list = OVO(data,[1,2,3],"SLP")
# w_list = OVO(data,[1,2,3],"Pocket")
# w_list = OVO(data,[1,2,3],"Adaline",[0.1])
# w_list = OVO(data,[1,2,3],"Logistic",[0.05, 500, 0.1])
# w_list = OVO(data,[1,2,3],"Pocket Trans",[5])
# w_list = OVO(data,[1,2,3],"Adaline Trans",[5])

t = "OVA Pocket"

# calculating loss
y_pred = predict(data[:,:data.shape[1]-1], w_list)
los = loss(data[:,-1],y_pred)

# ploting results
import matplotlib.pyplot as plt

for i in range(len(data)):
    if data[i][-1] == 1 :
        plt.plot(data[i][1],data[i][2], "or")
    elif data[i][-1] == 2 :
        plt.plot(data[i][1],data[i][2], "og")
    else :
        plt.plot(data[i][1],data[i][2], "ob")

Z = [i for i in range(-3,10)]

# hyperplan equation (in 2D)
def h(wop,x):
    return -(wop[1]*x + wop[0])/wop[2]
# hyperplan equation (in 2D) for +2 vars
def htr(wop,x):
    return sum([wop[i] * x[i] for i in range(len(wop))])

for w in w_list :
    # linear plot
    print(w)
    plt.plot(Z,[h(w, e) for e in Z],"-", color="black")
    # non linear plot
    #plt.plot(data,[htr(w, data, 5) for i in range(data.shape[0])],"-", color="black")


# limuts of separable data
plt.xlim([-4, 12])
plt.ylim([-4, 14])
plt.title(t +" on separable data\nloss = "+str(los))

# limits of noisy data
# plt.xlim([-3, 7])
# plt.ylim([-3, 7])
plt.show()
plt.savefig(t)