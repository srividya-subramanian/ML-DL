# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:57:28 2024

@author: srivi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from keras.utils import to_categorical 
from math import e 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def perceptron(x, w): 
    y = np.dot(x, w)
    return y

def activationfn(y):
    sig = 1 / (1+e**(-y))
    return sig

def derivativefn(y):
    dsig = e**(-y)/ (1+e**(-y))
    return dsig

def initialise():
    inputLayerSize, hiddenLayerSize, outputLayerSize = 784,100,2
    wi = np.random.rand(inputLayerSize, hiddenLayerSize) 
    wo = np.random.rand(hiddenLayerSize, outputLayerSize) 
    return wi, wo

def forward_propagation(x,wi,wo):
    hx = perceptron(x,wi)
    h  = activationfn(hx)
    hy = perceptron(h,wo)
    y_pred = activationfn(hy)
    return y_pred, h

def error(y, y_pred):
    err = y - y_pred
    return err

def back_propagation(h, wi, wo, x, y, y_pred,  lr):
    err = error(y, y_pred)
    dy_pred = err * derivativefn(y_pred)
    errh = np.dot(dy_pred, wo.T)
    dh = errh * derivativefn(h)
    
    wi += np.dot(x.T, dh)*lr
    wo += np.dot(h.T,y_pred)*lr
    return wi, wo


df= pd.read_csv('mnist_train.csv')
dftest=pd.read_csv('mnist_test.csv')

df0=df.loc[df['label']==0]
df1=df.loc[df['label']==1]
dftest0=dftest.loc[dftest['label']==0]
dftest1=dftest.loc[dftest['label']==1]

train  = pd.concat([df0,df1],axis=0, ignore_index=True)
train = train.sample(frac=1).reset_index(drop=True)
test  = pd.concat([dftest0,dftest1],axis=0, ignore_index=True)
test = test.sample(frac=1).reset_index(drop=True)

dfX_train = train.drop(['label'],axis=1)
dfY_train = train[['label']]
dfX_test = test.drop(['label'],axis=1)
dfY_test = test[['label']]

X_train = dfX_train.values
Y_train = dfY_train.values
X_test = dfX_test.values
Y_test = dfY_test.values
#X_train = X_train.reshape((X_train.shape[0], 28, 28))
#Y_trainY = to_categorical(Y_train)


pixel = dfX_train.iloc[0]
greyscale = [pixel if pixel > 120 else 0 for pixel in pixel]
bildArray = np.array(pixel, dtype=int).reshape((28, 28)) #7

plt.imshow(bildArray, cmap='Greys') #8
plt.show() #9

wi, wo = initialise()
lr = 0.3 
epochs = 1000
for epoch in range(epochs): 
   y_pred, h = forward_propagation(X_train , wi, wo) 
   wi, wo = back_propagation(h, wi, wo, X_train, Y_train, y_pred, lr) 


error = error(Y_train, y_pred)
print(Y_train[0:20], y_pred[0:20])
yy, h = forward_propagation(X_test, wi, wo) 


Y_pred = [0] * len(Y_test)
    
for i in range(len(Y_test)):
   if (np.mean(yy) > 0.5): 
       Y_pred[i] = 1
   elif (np.mean(yy) <= 0.5):
       Y_pred[i] = 0
       
accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)#, pos_label='0')
recall = recall_score(Y_test, Y_pred)#, pos_label='0')
f1score = f1_score(Y_test, Y_pred)#, pos_label='0')

print(accuracy, precision, recall, f1score)













