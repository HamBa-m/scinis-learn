# Perceptron, Pocket, Adaline, Logistic regression and Polynomial transformation with Pocket and Adaline.
import numpy as np
from algos.perceptron import PLA
from algos.pocket import Pocket
from algos.adaline import Adaline
from algos.logistic_regression import LogisticRegression
from algos.nonLinearTransformer import *

def OVA(data, classes, algo, hyperpara=[]):
    w_list = list()
    for e in classes:
        data_ = np.copy(data)
        row = data_.shape[0]
        for j in range(row):
            if int(data_[j][-1]) != e : data_[j][-1] = -1
            else : data_[j][-1] = 1
        if algo =="SLP":
            # hyperpara = []
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            w, t = PLA(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Pocket":
            # hyperpara = []
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Adaline":
            # hyperpara = [eps]
            w = np.zeros(data.shape[1] - 1)
            col = data_.shape[1]
            eps = np.copy(hyperpara[0])
            w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
            w_list.append(w)
        elif algo == "Logistic":
            # hyperpara = [lr, Tmax, epsilon]
            col = data_.shape[1]
            lr = np.copy(hyperpara[0])
            Tmax = np.copy(hyperpara[1])
            eps = np.copy(hyperpara[2])
            w, t, ls = LogisticRegression(data_[:,:col-1],data_[:,-1],lr,Tmax,eps)
            w_list.append(w)
        elif algo == "Pocket Trans":
            # hyperpara = [q]
            q = np.copy(hyperpara[0])
            data_ = np.asarray([psy(x,q) for x in data_])
            w = np.zeros(data_.shape[1] - 1)
            col = data_.shape[1]
            w, t, ls = Pocket(data_[:,:col-1],data_[:,-1],w)
            w_list.append(w)
        elif algo == "Adaline Trans":
            # hyperpara = [q]
            q = np.copy(hyperpara[0])
            data_ = np.asarray([psy(x,q) for x in data_])
            w = np.zeros(data_.shape[1] - 1)
            col = data_.shape[1]
            eps = np.copy(hyperpara[0])
            w, t, ls = Adaline(data_[:,:col-1],data_[:,-1],w,eps)
            w_list.append(w)
    return w_list