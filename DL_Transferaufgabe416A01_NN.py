# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:08:51 2025

@author: srivi
"""
import numpy as np


def predict(x,w):
    pred = np.dot(x,w)
    return pred

def error(y, y_pred):
    err = y - y_pred
    return err

def gradw(x, y, y_pred):
    sum = 0
    n = len(y)
    for i in range (0,n):
        err = y_pred[i] - y[i]
        sum += -x[i] * err
    dw =   sum 
    #print('delw', dw)
    return dw

X = [0.5]
Y = [0.8]
w = 0.5
lr = 0.1

for j in range(0,20):
    y_pred = predict(X, w)
    err = error(Y, y_pred)
    dw = gradw(X, Y, y_pred)
    w_new = w - dw *lr
    w=w_new
    print(err, y_pred)
  
 