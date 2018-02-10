import os
import sys
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



X = genfromtxt('dataset/linearX.csv',delimiter = ',') # list of training example vectors
X_save = X
if X.ndim == 1:
    X = X[np.newaxis]
    X = np.transpose(X) # convert into 2-d matrix if there is only one feature
std_dev =  np.std(X,axis=0)
mean = np.mean(X,axis=0)
mean = np.tile(mean,(len(X),1))
std_dev = np.tile(std_dev,(len(X),1))

X = (X - mean)/std_dev


Y = genfromtxt('dataset/linearY.csv',delimiter = ',') # list of training outputs
Y_save = Y
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
        if count_iter%10==1:
            ax.scatter(theta[0],theta[1],J_theta[0][0],marker='o',c='r')
        print "iteration count -> ", count_iter, "\t J(theta) -> ",np.asscalar(J_theta)
        if abs(np.asscalar(J_theta - prev_J_theta)) < epsilon:
            return theta
        prev_J_theta = J_theta
        X_transpose = np.transpose(X)
        update_theta = np.matmul(X_transpose,error) #using linear derivative formulae
        theta = theta + alpha * update_theta


def normal_eqns():
    global X,Y
    Xt = np.transpose(X)
    Xt_X = np.matmul(Xt,X)
    Xt_X_inv = np.linalg.inv(Xt_X)
    Xt_X_inv_Xt = np.matmul(Xt_X_inv,Xt)
    theta = np.matmul(Xt_X_inv_Xt,Y)
    return theta


fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
# theta = gradient_descent()
theta1 = np.arange(-0.5,1,0.005)
theta2 = np.arange(0,1,0.005)
theta1, theta2 = np.meshgrid(theta1, theta2)
for i in range(m):
    if i==0:
        Z = (theta1 + theta2 * X[i][1] - Y[i][0])**2
    else:
        Z += (theta1 + theta2 * X[i][1] - Y[i][0])**2
Z = Z /2.0


# Plot the surface.
surf = ax.plot_surface(theta1, theta2, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 50)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

theta = gradient_descent()
# Add a color bar which maps values to colors.
# fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#

# theta = normal_eqns()
# X_min = np.min(X_save)
# X_max = np.max(X_save)
# Y_min = theta[0][0] + theta[1][0] * X_min
# Y_max = theta[0][0] + theta[1][0] * X_max
# plt.plot([X_min,X_max],[Y_min,Y_max],c='b')
# plt.plot(X_save, Y_save,'ro')
# plt.show()
