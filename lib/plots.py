import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
import numpy as np

from .transformation import psy
from .hypothesis import h
from .activation import sigmoid

# hyperplan equation in 2D
def line(wop,x):
    return -(wop[1]*x + wop[0])/wop[2]

# hyperplan equation in 3D
def surface(wop,x,y):
    return -(wop[1] * x + wop[2] * y + wop[0])/wop[3]

# regression curve in 2D
def curve(wop,x):
    return sum([wop[i] * x for i in range(len(wop))])


def plotLinearDecisionBoundary2D(X, Y, w0):
    """
    description: plot linear decision boundary in 2 dimensions with matplotlib
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w0: weight vector
    return: None   
    """
    for i in range(len(X)):
        if Y[i] == 1 :
            plt.plot(X[i][1],X[i][2], "or")
        else :
            plt.plot(X[i][1],X[i][2], "og")
            
    Z = [i for i in range(round(min(X[:,1])), round(max(X[:,1])) +1)]

    hyper = [line(w0, e) for e in Z]
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Plot of linear decision boundary")
    plt.plot(Z,hyper,"-b")
    plt.legend()
    plt.show()

def plotLinearDecisionBoundary3D(X, Y, w0):
    # Create the figure
    fig = plt.figure()
    # Add an axes
    ax = fig.add_subplot(111,projection='3d')
    for e in range(len(X)):
        if Y[e] == 1 :ax.plot(X[e][1], X[e][2], 1, "o", color = 'red')
        else : ax.plot(X[e][1], X[e][2], 0, "o", color = 'green')

    x, y = np.arange(-4,2,0.1),np.arange(-4,2,0.1)
    x, y = np.meshgrid(x,y)
    z = surface(w0, x, y)
    surf = ax.plot_surface(x, y, z,cmap=cm.coolwarm,
                        linewidth=0, antialiased=False, alpha= 0.6)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("x3")
    ax.set_title("Plot of Linear Decision Boundary")

    plt.show()


# plot non linear decision boundary in 2 dimensions with matplotlib
def plotNonLinearDecisionBoundary2D(X, Y, w, q):
    """
    description: plot non linear decision boundary in 2 dimensions with matplotlib
    args:
        X: a list of vectors x_i
        Y: the list of labels y_i associated
        w: weight vector
        q: polynom's degree
    return: None
    """

    # create meshgrid
    print(X.shape)
    x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
    z = (x_max - x_min)/100 
    xx, yy = np.meshgrid(np.arange(x_min, x_max, z), np.arange(y_min, y_max, z))

    # create color map
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

    # create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    # plot decision surface
    Z = np.array([sigmoid(h(psy(np.array([x,y]),q), w)) for x, y in zip(np.ravel(xx), np.ravel(yy))])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm.viridis, alpha=0.8)
    ax.contour(xx, yy, Z, [0.5], linewidths=2, colors='k')

    # plot training points
    ax.scatter(X[:, 1], X[:, 2], c=Y, cmap=cmap_bold, edgecolor='k', s=20)

    # set labels
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title("Non linear decision boundary\ndegree = " + str(q))
    # add colorbar
    fig.colorbar(ax.contourf(xx, yy, Z, cmap=cm.viridis, alpha=0.5), ax=ax, label="Probability")

    plt.show()
