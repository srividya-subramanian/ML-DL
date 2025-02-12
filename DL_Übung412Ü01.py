# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:35:12 2025

@author: srivi
"""

import numpy as np
import pandas as pd

X = [1,2,3,4,5]                 #studienstunde
Y = [2.0, 2.9, 3.8, 4.2, 5.0]   #abschlossnote
n = len(Y)

w = 0                           #weight
b = 0                           #bias
lr = 0.01 

def neural_network(x,w,b):
    pred = np.dot(x,w)+b
    return pred

def mse(y, y_pred):
    n = len(y)
    sum = 0
    for i in range (0,n):
        sum += (y[i] - y_pred[i]) ** 2
    mse = sum / n
    #print('MSE',mse)
    return mse

def gradw(x, y, y_pred):
    sum = 0
    n = len(y)
    for i in range (0,n):
        err =y[i] - y_pred[i]
        sum += -x[i] * err
    dw =  (2/n) * sum 
    #print('delw', dw)
    return dw
         
def gradb(x, y, y_pred):
    sum = 0
    n = len(y)
    for i in range (0,n):
        err =y[i] - y_pred[i]
        sum += - err
    db = (2/n) * sum               
    #print('delb', db)
    return db

for j in range(0,100):
    y_pred = neural_network(X, w, b)
    err = mse(Y, y_pred)
    dw = gradw(X, Y, y_pred)
    db = gradb(X, Y, y_pred)
    w_new = w - lr*dw
    b_new = b - lr*db
    print(w_new, b_new, err)
    w=w_new
    b=b_new



