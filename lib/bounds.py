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


def Ne(data, epsilon, d = 2):
    """
    function that computes the number of balls of radius epsilon that cover the data
    args:
        data: a list of points
        epsilon: the radius of the balls
        d: the norm used to compute the distance between points
    returns:
        ne: the number of balls of radius epsilon that cover the data
    """
    points  = list(data)
    for i in range(len(points)):
        points[i] = list(points[i])
        points[i].append(0)
    ne = 0
    n_cov = 0
    while n_cov < len(points):
        current_cov = list()
        for p in points:
            voisinage_p = [other for other in points if (np.linalg.norm(np.asarray(other[:-1]) - np.asarray(p[:-1]), ord = d) <= epsilon) and (other[-1] != 1)]
            current_cov.append([p, len(voisinage_p), voisinage_p])
        current_cov.sort(key=lambda x: x[1], reverse=True)
        for e in points:
            if e in current_cov[0][2]:
                e[-1] = 1
        n_cov += current_cov[0][1]
        ne += 1
    return ne

def uniform_covering_numbers(classifier, radius, m, data, d = 2):
    """
    function that computes the uniform covering number of the data
    args:
        classifier: the classifier used to compute the covering number
        radius: the radius of the balls
        m: the number of points in the subset
        data: the data
        d: the norm used to compute the distance between points
        returns:
            max_cov_nbr: the uniform covering number of the data
    """
    A = np.asarray(list(itertools.combinations(data, m)))
    max_cov_nbr = 0
    for subset in A:
        classifier.fit(subset[:, :-1], subset[:, -1])
        H_subset = classifier.predict(subset[:, :-1])
        for i in range(subset.shape[0]):
            subset[i][-1] = H_subset[i]
        cov_nbr = Ne(subset, radius, d)
        print("cov_nbr = ", cov_nbr)
        if cov_nbr > max_cov_nbr:
            max_cov_nbr = cov_nbr

    return max_cov_nbr