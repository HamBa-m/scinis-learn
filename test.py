# from lib.data import generateData

# # make the list of circles as dictionnaries with keys 'center' and 'radius' and "class"
# circles = [
#     {'center': [0, 0], 'radius': 1, 'class': 1},
#     {'center': [2, 2], 'radius': 1, 'class': 0},
#     # {'center': [-2, -2], 'radius': 1, 'class': 1},
#     # {'center': [2, -2], 'radius': 1, 'class': -1},
#     # {'center': [-2, 2], 'radius': 1, 'class': 1},
#     ]

# # Generate 500 points in 2D
# # data = generateData(500, circles, filename='data2Dlogistic.csv')


# # ### Perceptron Algorithm

# # import numpy as np
# # from lib.classification import PLA
# # from lib.data import loadData

# # # Load data
# # data = loadData('data2D.csv', ['x0','x1', 'x2', 'y'])
# # X = data[:, :-1]
# # y = data[:, -1]

# # # Initialize the weights vector
# # w = np.zeros(X.shape[1])

# # # Initialize the Perceptron Algorithm
# # w0, t = PLA(X, y, w)

# # # Print the weights vector and the number of iterations
# # print(w0, t)


# # ## Plotting Decision Boundary (in 2D)


# # from lib.plots import plotLinearDecisionBoundary2D

# # # Plot the decision boundary
# # plotLinearDecisionBoundary2D(X, y, w0)


# # logistic regression algorithm

# from lib.classification import LogisticRegression
# from lib.data import loadData
# import numpy as np

# # Load data
# data = loadData('data2Dlogistic.csv', ['x0','x1', 'x2', 'y'])
# X = data[:, :-1]
# y = data[:, -1]

# # Initialize the logistic regression algorithm
# w0, loss = LogisticRegression(X, y)

# # Print the weights vector and the number of iterations
# print(w0, loss)

# # Plot the decision boundary
# from lib.plots import plotNonLinearDecisionBoundary2D
# plotNonLinearDecisionBoundary2D(X, y, w0, 1)

# try linear regression

# # make linear regression data
# from lib.data import makeRegressionData
# data = makeRegressionData(100, degree=1)

# # load data
# # from lib.data import loadData
# # data = loadData('data.csv', ['x0','x1', 'y'])
# X = data[:, :-1]
# y = data[:, -1]

# # initialize linear regression
# from lib.regression import LinearRegression
# w0, loss = LinearRegression(X, y)

# # print the weights vector and the loss
# print(w0, loss)

# # plot the decision boundary
# from lib.plots import plotRegressionLine2D
# plotRegressionLine2D(X, y, w0)

# make polynomial regression data
from lib.data import makeRegressionData
data = makeRegressionData(100, degree=3)

X = data[:, :-1]
y = data[:, -1]

# transform the data to a higher dimension
from lib.transformation import polyMap
X3 = polyMap(X, 3)

# initialize polynomial regression
from lib.regression import LinearRegression
w0, loss = LinearRegression(X3, y)

# print the weights vector and the loss
print(w0, loss)
