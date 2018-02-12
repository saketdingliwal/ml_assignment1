import os
import sys
import numpy as np
from numpy import genfromtxt
import time
import matplotlib.pyplot as plt

alpha = 0.0008 # learning rate
epsilon = 1e-10 # stopping criterion


def get_data():
    X = genfromtxt('dataset/logisticX.csv',delimiter = ',') # list of training example vectors
    X_save = X
    if X.ndim == 1:
        X = X[np.newaxis]
        X = np.transpose(X) # convert into 2-d matrix if there is only one feature
    Y = genfromtxt('dataset/logisticY.csv',delimiter = ',') # list of training outputs
    Y_save = Y
    Y = Y[np.newaxis]
    Y = np.transpose(Y) # now we have Y as column vector
    return X,Y,X_save,Y_save
    
#normalize data
def normalize(X):
    std_dev =  np.std(X,axis=0)
    if std_dev.any()==0:
        print "standard deviation of the data is zero"
    mean = np.mean(X,axis=0)
    mean = np.tile(mean,(len(X),1))
    std_dev = np.tile(std_dev,(len(X),1))
    X = (X - mean)/std_dev
    return X



def g_theta(z):
    denom = 1 + np.exp(-1 * z)
    return 1.0/denom

def newton_method():
    global X,Y,epsilon
    n = len(X[0]) # number of input features
    theta = np.zeros((n,1)) # initialized parameters with zero
    count_iter = 0
    while(True):
        count_iter +=1
        theta_t_x = np.matmul(X,theta)
        hypothesis_val = g_theta(theta_t_x)
        error = Y - hypothesis_val
        grad_l_theta = np.matmul(np.transpose(X),error)
        g_one_minus_g = hypothesis_val * (hypothesis_val - 1)
        to_diagonalize = np.transpose(g_one_minus_g)
        diag_matr = np.diag(to_diagonalize[0])
        X_transpose = np.transpose(X)
        hessian = np.matmul(np.matmul(X_transpose,diag_matr),X)
        if np.linalg.det(hessian)==0:
            print "the matrix is singular using pseudo inverse instead"
            hessian_inv = np.linalg.pinv(hessian)
        else:
            hessian_inv = np.linalg.inv(hessian)
        update_amount = np.matmul(hessian_inv,grad_l_theta)
        print "iteration count -> ", count_iter, "\t delta_theta -> ",np.linalg.norm(np.transpose(update_amount))
        # checking the convergence condition
        if np.linalg.norm(np.transpose(update_amount)) < epsilon:
            return theta
        theta = theta - update_amount


X,Y,X_save,Y_save = get_data()
X = normalize(X)

m = len(X) # number of training examples
if m == 0 :
    print "Training data missing"
x0 = np.ones((m,1))
X = np.hstack((x0,X)) # adding ones to training vectors

theta = newton_method()
print "theta",theta
# plot points
for i in range(m):
    if Y[i][0]==0:
        y0, = plt.plot(X[i][1],X[i][2],'ro',c='b',label='0')
    else:
        y1, = plt.plot(X[i][1],X[i][2],'ro',c='r',label='1')
plt.legend(handles=[y0,y1])
X_transpose = np.transpose(X)
X_min = np.min(X_transpose[1])
X_max = np.max(X_transpose[2])
x = np.linspace(X_min, X_max, 50)
plt.plot(x,(-1.0/theta[2])*(theta[0]+theta[1]*x),c='g')
plt.plot()
plt.savefig('logistic' + ".jpeg")
plt.show()
