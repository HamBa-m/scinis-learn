import itertools
import numpy as np
import matplotlib.pyplot as plt

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
        ##### plot the subset and the balls ####
        plt.scatter(subset[:, :-1], subset[:, -1])
        for e in subset:
            c = plt.Circle(e, radius, color='r', fill=True, alpha=0.2)
            plt.gca().add_patch(c)
        plt.grid()
        plt.show()
        ########################################
        cov_nbr = Ne(subset, radius, d)
        print("cov_nbr = ", cov_nbr)
        if cov_nbr > max_cov_nbr:
            max_cov_nbr = cov_nbr

    return max_cov_nbr