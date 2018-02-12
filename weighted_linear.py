import os
import sys
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt
import random

X = genfromtxt('dataset/weightedX.csv',delimiter = ',') # list of training example vectors
X_save = X
if X.ndim == 1:
    X = X[np.newaxis]
    X = np.transpose(X) # convert into 2-d matrix if there is only one feature

std_dev =  np.std(X,axis=0)
if std_dev.any()==0:
    print "standard deviation of the data is zero"
mean = np.mean(X,axis=0)
mean = np.tile(mean,(len(X),1))
std_dev = np.tile(std_dev,(len(X),1))

X = (X - mean)/std_dev

Y = genfromtxt('dataset/weightedY.csv',delimiter = ',') # list of training outputs
Y_save = Y
Y = Y[np.newaxis]
Y = np.transpose(Y) # now we have Y as column vector

m = len(X) # number of training examples
if m == 0 :
    print "Training data missing"
x0 = np.ones((m,1))
X = np.hstack((x0,X)) # adding ones to training vectors

alpha = 0.00004 # learning rate
epsilon = 0.0000001 # stopping criterion
tau = 0.3


def make_W(point_X):
    global X,Y,tau
    m = len(X)
    W = np.zeros((m,m))
    n = len(X[0])
    covar_inv = np.zeros((n,n))
    np.fill_diagonal(covar_inv,1.0/(tau*tau))
    for i in range(m):
        matrix = np.matmul(np.transpose(X[i] - point_X),covar_inv)
        matrix = np.matmul(matrix,(X[i] - point_X))
        W[i][i] = np.exp(-0.5*matrix)
    return W

def normal_eqns():
    global X,Y
    Xt = np.transpose(X)
    Xt_X = np.matmul(Xt,X)
    if np.linalg.det(Xt_X)==0:
        print "the matrix is singular using pseudo inverse instead"
        Xt_X_inv = np.linalg.pinv(Xt_X)
    else:
        Xt_X_inv = np.linalg.inv(Xt_X)
    Xt_X_inv_Xt = np.matmul(Xt_X_inv,Xt)
    theta = np.matmul(Xt_X_inv_Xt,Y)
    return theta

def normal_wtd_eqns(point_X):
    global X,Y
    Xt = np.transpose(X)
    W = make_W(point_X)
    Xt_W = np.matmul(Xt,W)
    Xt_W_X = np.matmul(Xt_W,X)
    if np.linalg.det(Xt_W_X)==0:
        print "the matrix is singular using pseudo inverse instead"
        Xt_W_X_inv = np.linalg.pinv(Xt_W_X)
    else:
        Xt_W_X_inv = np.linalg.inv(Xt_W_X)
    Xt_W_X_inv_Xt = np.matmul(Xt_W_X_inv,Xt)
    Xt_W_X_inv_Xt_W = np.matmul(Xt_W_X_inv_Xt,W)
    theta = np.matmul(Xt_W_X_inv_Xt_W,Y)
    return theta


# theta = gradient_descent()
theta = normal_eqns()
print theta
# plotting the unweighted line
X_save = (X_save-np.mean(X_save))/np.std(X_save)
X_min = np.min(X_save)
X_max = np.max(X_save)
Y_min = theta[0][0] + theta[1][0] * X_min
Y_max = theta[0][0] + theta[1][0] * X_max
plt.plot([X_min,X_max],[Y_min,Y_max],c='g')
plt.plot(X_save, Y_save,'ro')

# plotting the weighted line on data
X_save_for_plot = []
for i in range(1000):
    X_save_for_plot.append(random.uniform(X_min,X_max))
X_save_sort = np.sort(X_save_for_plot)
Hyp_save = []
for i in range(1000):
    point_X = np.zeros(2)
    point_X[0] = 1
    point_X[1] = X_save_sort[i]
    theta = normal_wtd_eqns(point_X)
    Hyp_save.append(theta[0][0] + theta[1][0] * X_save_sort[i])
plt.plot(X_save_sort,Hyp_save,c="b")
plt.savefig('weighted_linear' + (str)(tau) + ".jpeg")
plt.show()

