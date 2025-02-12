# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 15:58:57 2025

@author: srivi
"""

import numpy as np

Feature1 = [5.1, 7.0, 4.6, 6.5, 5.0, 5.7]
Feature2 = [3.5, 3.2, 3.1, 2.8, 3.6, 2.8]
Feature3 = [1.4, 4.7, 1.5, 4.6, 1.4, 4.5]
Feature4 = [0.2, 1.4, 0.2, 1.5, 0.3, 1.2]
Art = [0,1,0,1,0,1]

X = np.array([Feature1, Feature2, Feature3, Feature4]).T
y = np.array(Art).T.reshape(6,1)

def initialise():
    ipLayerSize, hdLayerSize, opLayerSize = 4,32,1
    wi = np.random.rand(ipLayerSize, hdLayerSize) 
    bi = np.zeros((1, hdLayerSize))
    wo = np.random.rand(hdLayerSize, opLayerSize) 
    bo = np.zeros((1, opLayerSize))
    print(wi.shape, wo.shape, bi.shape, bo.shape)
    return wi, wo, bi, bo


def perceptron_npdot(x, w): 
    y = np.dot(x, w)
    return y

def sigmoid_activationfn(y):
    sig = 1 / (1+np.exp(-y))
    return sig

def derivative_sigmoid_fn(y):
    dsig = np.exp(-y)/ (1+np.exp(-y))
    return dsig

def error(y, y_pred):
    err = y - y_pred
    return err

def forward_propagation(x,wi,wo,bi,bo):
    hx = perceptron_npdot(x,wi)+bi
    ho  = sigmoid_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = (sigmoid_activationfn(yh))
    return y_pred, ho

def back_propagation(h, wi, wo, bi, bo, x, y, y_pred,  lr):
    err = error(y, y_pred)
    deltao = err * derivative_sigmoid_fn(y_pred)
    deltah = np.dot(deltao, wo.T) * derivative_sigmoid_fn(h)

    wo += np.dot(h.T, deltao)*lr
    wi += np.dot(x.T, deltah)*lr
    bo += np.sum(deltao, axis=0, keepdims=True) 
    bi += np.sum(deltah, axis=0, keepdims=True) 
    return wi, wo


wi, wo, bi, bo = initialise()
lr = 0.3 
epochs = 1000
for epoch in range(epochs): 

    y_pred, h = forward_propagation(X,wi,wo,bi,bo)
    wi, wo = back_propagation(h, wi, wo, bi, bo, X, y, y_pred, lr)

y_pred=y_pred.round(1).astype(int)
print(y)
print(y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)#, pos_label='0')
recall = recall_score(y, y_pred)#, pos_label='0')
f1score = f1_score(y, y_pred)#, pos_label='0')

print(accuracy, precision, recall, f1score)














