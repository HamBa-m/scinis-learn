import numpy as np
import itertools

def create_combi(n, arr, i, combi):
    """
    Create all possible combinations of 0 and 1 for the last column of the array
    args:
        n: the number of rows of the array
        arr: the array to create combinations
        i: the current row to create combinations
        combi: the list to store all combinations
    """
    if i == n:
        combi.append(np.copy(arr))
        return
    arr[i][-1] = 0
    create_combi(n, arr, i + 1, combi)
    arr[i][-1] = 1
    create_combi(n, arr, i + 1, combi)
    

def shatter(classifier, data):
    """
    Check if the classifier can shatter the data
    args:
        classifier: the classifier to check
        data: the data to check
    return: 
        True if the classifier can shatter the data
    """
    all_combi = []
    X = np.copy(data)
    create_combi(len(X), X, 0, all_combi)
    all_combi = np.asarray(all_combi)
    MAX = all_combi.shape[0]
    for i in range(MAX):
        if all(all_combi[i][:, -1] == 0) or all(all_combi[i][:, -1] == 1) :
            continue
        classifier.fit(all_combi[i][:, :-1], all_combi[i][:, -1])
        y_ = classifier.predict(all_combi[i][:, :-1])
        if not all(y_ == all_combi[i][:, -1]):
            return False
    return True

def VC_dimension(classifier, data):
    """
    Calculate the VC dimension of the classifier
    args:
        classifier: the classifier to calculate
        data: the data to calculate
    return:
        the VC dimension of the classifier
    """
    vc = 1
    for k in range(2, data.shape[0] + 1):
        A = list(itertools.combinations(data, k))
        i = 1
        for subset in A:
            if shatter(classifier, np.asarray(subset)):
                vc += 1
                break
            if i == len(A):
                return vc
            i += 1
    return vc