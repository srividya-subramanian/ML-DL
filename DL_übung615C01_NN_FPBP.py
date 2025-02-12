# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:43:36 2025

@author: srivi
"""

import numpy as np
import numpy.random as rand
import pandas as pd
from sklearn.model_selection import train_test_split
rand.seed(42)

'''Initialise random data'''
speed = rand.uniform(0,200,1000)
acceleration = rand.uniform(-5,5,1000)
braking_behavior=rand.randint(0, 2, 1000) 
steering_angle=rand.uniform(-45,45,1000)

data = np.column_stack((speed, acceleration, braking_behavior, steering_angle))
x = data

df = pd.DataFrame(data, columns=['speed', 'acceleration', 'braking_behavior', 'steering_angle'])
Y=df['braking_behavior']
X=df.drop(['braking_behavior'],axis=1)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=.25)
y_train = np.array(y_train).reshape(750,1)

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

def initialise():
    inputLayerSize, hiddenLayerSize, outputLayerSize = 3,32,1
    wi = np.random.rand(inputLayerSize, hiddenLayerSize) 
    bi = np.zeros((1, hiddenLayerSize))
    wo = np.random.rand(hiddenLayerSize, outputLayerSize) 
    bo = np.zeros((1, outputLayerSize))
    return wi, wo, bi, bo


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

    y_pred, h = forward_propagation(X_train,wi,wo,bi,bo)
    wi, wo = back_propagation(h, wi, wo, bi, bo, X_train, y_train, y_pred, lr)

y_pred=y_pred.round(1).astype(int)
#print(y)
#print(y_pred)
y = y_train
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
#precision = precision_score(y, y_pred)#, pos_label='0')
recall = recall_score(y, y_pred)#, pos_label='0')
f1score = f1_score(y, y_pred)#, pos_label='0')

print(accuracy, recall, f1score)
#print(accuracy, precision, recall, f1score)
   

