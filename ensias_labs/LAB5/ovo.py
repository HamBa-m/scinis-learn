# Perceptron, Pocket, Adaline, Logistic regression and Polynomial transformation with Pocket and Adaline.
import numpy as np
from algos.perceptron import PLA
from algos.pocket import Pocket
from algos.adaline import Adaline
from algos.logistic_regression import LogisticRegression
from algos.nonLinearTransformer import *

def OVO(data, classes, algo, hyperpara=[]):
    w_list = list()
    col = data.shape[1]
    for e in classes:
        print(classes,e)
        classes.remove(e)
        for f in classes:
            data_ = []
            for k in range(data.shape[0]):
                if data[k][-1] == e : 
                    l = list(data[k,:col - 1])
                    l.append(1)
                    data_.append(l)
                elif data[k][-1] == f : 
                    l = list(data[k,:col - 1])
                    l.append(-1)
                    data_.append(l)
            data_ = np.asarray(data_)
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