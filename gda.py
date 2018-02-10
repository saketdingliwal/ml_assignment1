import os
import sys
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt
import math

X = [i.strip().split() for i in open("dataset/q4x.dat").readlines()]
for i in range(len(X)):
    for j in range(len(X[0])):
        X[i][j] = float(X[i][j])
X = np.array(X)
X_save = X
if X.ndim == 1:
    X = X[np.newaxis]
    X = np.transpose(X) # convert into 2-d matrix if there is only one feature

std_dev =  np.std(X,axis=0)
mean = np.mean(X,axis=0)
mean = np.tile(mean,(len(X),1))
std_dev = np.tile(std_dev,(len(X),1))

X = (X - mean)/std_dev


m = len(X) # number of training examples
if m == 0 :
    print "Training data missing"



Y = [i.strip().split() for i in open("dataset/q4y.dat").readlines()]# list of training outputs
y = np.zeros(m)
for i in range(len(X)):
    y[i] = (Y[i] == ['Alaska'])
Y = y
Y = np.array(Y)
Y_save = Y
Y = Y[np.newaxis]
Y = np.transpose(Y) # now we have Y as column vector
y_tile = np.tile(Y,(len(X[0])))
# print y_tile

def gda():
    global X,Y
    y_one_count = np.sum(Y)
    y_zero_count = m - y_one_count
    phi = y_one_count/m
    u1 = np.matmul(np.transpose(Y),X) / (1.0 * y_one_count)
    u0 = np.matmul(np.transpose(1-Y),X)/y_zero_count
    u_y = np.matmul(Y,u1) + np.matmul((1-Y),u0)
    X_uy = X - u_y
    cov = np.matmul(np.transpose(X_uy),X_uy)/m
    X_uy_one = y_tile  * X_uy
    X_uy_zero = (1 - y_tile) * X_uy
    cov1 = np.matmul(np.transpose(X_uy_one),X_uy_one)/y_one_count
    cov0 = np.matmul(np.transpose(X_uy_zero),X_uy_zero)/y_zero_count
    # print phi,u0,u1,cov0,cov1,cov
    return phi,u0,u1,cov0,cov1,cov




phi,u0,u1,cov0,cov1,cov = gda()
for i in range(m):
    if Y[i][0]==0:
        plt.plot(X[i][0],X[i][1],'ro',c='b')
    else:
        plt.plot(X[i][0],X[i][1],'ro',c='r')

cov_inv = np.linalg.inv(cov)
cov_u1 = np.matmul(cov_inv,np.transpose(u1))
u1_cov_u1 = np.matmul(u1,cov_u1)
cov_u0 = np.matmul(cov_inv,np.transpose(u0))
u0_cov_u0 = np.matmul(u0,cov_u0)
theta = [0,0,0]
theta[0] = 2 * math.log(1-phi) - 2 * math.log(phi) + u1_cov_u1[0][0] - u0_cov_u0[0][0]
theta[1] = 2 * cov_u0[0][0] - 2 * cov_u1[0][0]
theta[2] = 2 * cov_u0[1][0] - 2 * cov_u1[1][0]
X_transpose = np.transpose(X)
X_min = np.min(X_transpose[0])
X_max = np.max(X_transpose[0])
x = np.linspace(X_min, X_max, 50)
plt.plot(x,(-1.0/theta[2])*(theta[0]+theta[1]*x),c='g')
plt.plot()
plt.show()
