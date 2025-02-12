# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 05:51:37 2025

@author: srivi
"""
from os import listdir,makedirs
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np

from numpy import load
photos = load('dogs_vs_cats_photos2_56x56.npy')
labels = load('dogs_vs_cats_labels2_56x56.npy')
print(photos.shape, labels.shape)

X = photos.reshape(25000,3136)
y = labels.reshape(25000,1)

def initialise():
    ipLayerSize, hdLayerSize, opLayerSize = 3136,32,1
    wi = np.random.rand(ipLayerSize, hdLayerSize) 
    bi = np.zeros((1, hdLayerSize))
    wo = np.random.rand(hdLayerSize, opLayerSize) 
    bo = np.zeros((1, opLayerSize))
    print(wi.shape, wo.shape, bi.shape, bo.shape)
    return wi, wo, bi, bo


def perceptron_npdot(x, w): 
    y = np.dot(x, w)
    return y

def relu_activationfn(x):
    if x<0:
        return 0
    else:
        return x

def derivative_relu(x): 
    if x<0:
        return 0
    else:
        return 1

def softmax_activationfn(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

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
    print(x.shape,wi.shape)
    print(((perceptron_npdot(x,wi))+bi).shape)
    hx = perceptron_npdot(x,wi)+bi
    ho  = relu_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = (softmax_activationfn(yh))
    return y_pred, ho


def back_propagation(h, wi, wo, bi, bo, x, y, y_pred,  lr):
    err = error(y, y_pred)
    deltao = err * derivative_relu(y_pred)
    deltah = np.dot(deltao, wo.T) * derivative_relu(h)

    wo += np.dot(h.T, deltao)*lr
    wi += np.dot(x.T, deltah)*lr
    bo += np.sum(deltao, axis=0, keepdims=True) 
    bi += np.sum(deltah, axis=0, keepdims=True) 
    return wi, wo


wi, wo, bi, bo = initialise()
lr = 0.3 
epochs = 1000
for epoch in range(epochs): 
    print(X.shape)
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



















