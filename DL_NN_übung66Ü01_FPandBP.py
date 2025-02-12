# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:47:13 2025

@author: srivi
"""

import numpy as np
from sklearn.datasets import make_biclusters

data, labels, cols = make_biclusters(shape =(10,2), n_clusters=2, random_state=42)


ii = []
jj = []
kk = []
for i,j in data:
    if (i > 0): i=1
    ii.append(i)
    if (j > 0): j=1
    jj.append(j)
    k=1
    kk.append(k)
data_new = [ii,jj,kk]
x = np.array(data_new).astype(int).T

y = labels.astype(int).T

w1 = np.random.rand(3,3)
w2 = np.random.rand(3,2)
epoch = 1000

def perceptron_npdot(x, w): 
    return np.dot(x, w)
def sigmoid(x): 
    return 1 / (1 + np.exp(-x))  
def sigmoid2deriv(x): 
    return x * (1 - x) 
 
def error(y, y_pred):
    return (y - y_pred)** 2
def delta(y, y_pred):
    return (y - y_pred)

def forward_pass(x,y,w1,w2):
    hid_layer = perceptron_npdot(x, w1) # hidden layer
    hid_output = sigmoid(hid_layer)
    op_layer = perceptron_npdot(hid_output, w2) #op layer
    output = sigmoid(op_layer)
    return output, hid_output

def backward_propagation(y, y_pred,y_hid):
    ypred_delta = delta(y, y_pred)*sigmoid2deriv(y_pred)
    dw2 = ypred_delta
    hidlayer_delta = perceptron_npdot(ypred_delta, w2.T) * sigmoid2deriv(y_hid)
    dw1 = hidlayer_delta
    return dw1, dw2

terr=[]
for i in range(epoch):
    y_pred, y_hid = forward_pass(x, y,w1,w2)
    err = error(y, y_pred)
    terr.append(np.sum(err))
    dw1,dw2 = backward_propagation(y, y_pred,y_hid)
    w1 += perceptron_npdot(x.T, dw1)
    w2 += perceptron_npdot(y_hid.T, dw2)
total_error = np.sum(err**2).round(7) 

print(w1)
print(w2)
print(total_error)
    
    
    
    
    