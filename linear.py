import os
import sys
import numpy as np
from numpy import genfromtxt
import time


X = genfromtxt('dataset/linearX.csv',delimiter = ',') # list of training example vectors
if X.ndim == 1:
    X = X[np.newaxis]
    X = np.transpose(X) # convert into 2-d matrix if there is only one feature

Y = genfromtxt('dataset/linearY.csv',delimiter = ',') # list of training outputs
Y = Y[np.newaxis]
Y = np.transpose(Y) # now we have Y as column vector

m = len(X) # number of training examples
if m == 0 :
    print "Training data missing"
x0 = np.ones((m,1))
X = np.hstack((x0,X)) # adding ones to training vectors

alpha = 0.00004 # learning rate
epsilon = 0.0001 # stopping criterion



def gradient_descent():
    global X,Y,alpha,epsilon
    n = len(X[0]) # number of input features
    theta = np.zeros((n,1)) # initialized parameters with zero
    prev_J_theta = np.zeros((1,1))
    count_iter = 0
    while(True):
        count_iter +=1
        hypothesis_val = np.matmul(X,theta)
        error = Y - hypothesis_val
        J_theta = 0.5 * np.matmul(np.transpose(error),error)
        print "iteration count -> ", count_iter, "\t J(theta) -> ",np.asscalar(J_theta)
        if abs(np.asscalar(J_theta - prev_J_theta)) < epsilon:
            return theta
        prev_J_theta = J_theta
        X_transpose = np.transpose(X)
        update_theta = np.matmul(X_transpose,error) #using linear derivative formulae
        theta = theta + alpha * update_theta


theta = gradient_descent()
