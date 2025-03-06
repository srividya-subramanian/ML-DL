# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:57:28 2024

@author: srivi
Develop a Python program that implements a simple three-layer neural network for
 the MNIST dataset. This network should consist of an input layer, a hidden layer 
 with ReLU activation function and an output layer. Integrate a regularization 
 technique to counteract overfitting. Use the L2 norm for the regularization and
 add it to the error term. In addition, the program should use batch gradient 
 descent to train the mesh. The batch size should be freely selectable. After 
 training, test the network with a separate test data set and output the accuracy 
 of the network on this test data. Comment your code to explain the individual 
 steps and the implementation of the regularization.  


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rand
from keras.utils import to_categorical 
from math import e 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def perceptron_npdot(x, w): 
    y = np.dot(x, w)
    return y

# Activation functions

def relu_activationfn(x):
    return np.maximum(x, 0)

def softmax_activationfn(x):
    exp_x = e**(x - np.max(x, axis=1, keepdims=True))
    y = exp_x/np.sum(exp_x)
    return y

def sigmoid_activationfn(y):
    sig = 1 / (1+e**(-y))
    return sig

# Derivative of Activation functions

def derivative_sigmoid_fn(y):
    dsig = np.exp(-y)/ (1+np.exp(-y))
    return dsig

def derivative_relu(x): 
    return x > 0

# Estimation of difference between the prediction and target
def error(y, y_pred):
    err = y - y_pred
    return err

# Cross-entropy loss with L2 regularization
def cross_entropy_loss(y_pred, y, wi, wo):
    loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
    l2_penalty = (l2_lambda / 2) * (np.sum(wi**2) + np.sum(wo**2))  # L2 Regularization term
    return loss + l2_penalty

# Initialize weights and biases
def initialise():
    inputLayerSize, hiddenLayerSize, outputLayerSize = 784,128,10
    wi = np.random.rand(inputLayerSize, hiddenLayerSize)* 0.01
    bi = np.zeros((1, hiddenLayerSize))
    wo = np.random.rand(hiddenLayerSize, outputLayerSize)* 0.01 
    bo = np.zeros((1, outputLayerSize))
    return wi, wo, bi, bo


def forward_propagation(x,wi,wo,bi,bo):
    hx = perceptron_npdot(x,wi)+bi
    h  = relu_activationfn(hx)
    hy = perceptron_npdot(h,wo)+bo
    y_pred = softmax_activationfn(hy)
    return y_pred, h


def back_propagation(h, wi, wo, x, y, y_pred,  lr):
    err = error(y, y_pred)
    dy_pred = err * derivative_relu(y_pred)
    errh = np.dot(dy_pred, wo.T)
    dh = errh * derivative_relu(h)
    #print(x.shape, dh.shape)
    wi += np.dot(x.T, dh)*lr
    wo += np.dot(h.T,dy_pred)*lr 
    return wi, wo

# Load MNIST dataset
df= pd.read_csv('C:/Users/srivi/Documents/ML_data/mnist_train.csv')
dftest=pd.read_csv('C:/Users/srivi/Documents/ML_data/mnist_test.csv')

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
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)

Y_train_onehot = np.eye(10)[Y_train]
Y_test_onehot = np.eye(10)[Y_test]


# Visualize data
pixel = dfX_train.iloc[0]
greyscale = [pixel if pixel > 120 else 0 for pixel in pixel]
bildArray = np.array(pixel, dtype=int).reshape((28, 28)) #7

plt.imshow(bildArray, cmap='Greys') #8
plt.show() #9

# Define Neural Network Parameters
wi, wo, bi, bo = initialise()
lr = 0.3 
epochs = 100
l2_lambda = 0.001
batch_size = 64

num_samples = X_train.shape[0]
num_batches = num_samples // batch_size

for epoch in range(epochs): 
   error, correct_cnt = (0.0, 0)
   shuffled_indices = np.random.permutation(num_samples)
   X_train_shuffled = X_train[shuffled_indices]
   y_train_shuffled = Y_train_onehot[shuffled_indices] 
   
   for batch in range(num_batches):
        # Extract mini-batch
        correct_cnt += int(np.argmax(layer_2[k:k+1]) ==
                           np.argmax(labels[batch_start+k:batch_start+k+1]))
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]
        y_pred, h = forward_propagation(X_batch , wi, wo, bi, bo) 
        correct_cnt += int(np.argmax(layer_2[k:k+1]) ==
                    np.argmax(labels[batch_start+k:batch_start+k+1]))        loss = cross_entropy_loss(y_pred, y_batch, wi, wo)
        wi, wo = back_propagation(h, wi, wo, X_batch, y_batch, y_pred, lr) 
        wo = wo/batch_size + (l2_lambda * wo)
        wi = wi/batch_size + (l2_lambda * wi)

print(Y_train[0:20], y_pred[0:20])
yy, h = forward_propagation(X_test, wi, wo, bi, bo) 


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




 








