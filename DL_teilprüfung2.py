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
from math import e


'''
Q: Explain how the introduction of conditional correlation and the use of the 
Rectified Linear Unit (ReLU) function improves the ability of a deep neural network 
to recognize complex patterns
Correlation: normally hidden middle layer is correlated with the input layer 
through weights - for example: x% to one input, y% to another and z% to the other.

Conditional correlation: makes the middle nodes be x% correlated to one input
when it wants to be and otherwise remain uncorrelated - for example: swithching
off nodes when it would be negative. By tuning the middle node like this is 
not possible in 2-layer neural network (NN). This logic makes the NN non-linear,
otherwise it would be linear. This one is the simplest form of introducing 
non-linearity and its called Relu (Rectified Linear Unit). The nonlinearity 
of activation functions is crucial for the success of predictive models and
enhance accuracy by enabling the system to learn parameters efficiently. The 
system learns to making conditional decisions.  

Use an example in which a neural network is trained 
to distinguish between images of cats and dogs. Include the following points:

a) Describe how a neural network without nonlinearities such as the ReLU function 
could only recognize linear patterns and why this is not sufficient for distinguishing 
between cats and dogs. 

If every layer of the network applies only linear transformations, then stacking 
multiple layers still results in a single equivalent linear transformation, making it 
equivalent to a single-layer model. It cannot learn complex decision boundaries.

Classification of cats and dogs is a nonlinear classification problem, with complex
features and patterns that cannot be seperated by a straight line


b) Explain how the ReLU function is implemented in the network and how it helps 
    to introduce nonlinearities. 
    
ReLU is implemented in a neural network in the hidden layers, after the linear 
transformation step, where it applies element wise transformation. 
y=max(0,x) --> if x > 0 then y = x, otherwise y = 0
It introduce non-linearities by breaking the linearity through setting negative 
y as 0, hence changing the shape of the linear function.

c) Explain how the conditional correlation is enabled by the ReLU function and why 
    this helps the network to learn more complex patterns such as the distinction
    between cats and dogs. 

In general correlation means, how neurons are correlated with each input nodes. 
Conditional correlation sets in and make the neurons correlated when certain 
conditions is satisfied and let them uncorrelated to the input nodes otherwise.
It refers to the idea of different parts of a neural network become active under 
different conditions, depending on the input data. 

Relu implements conditional correlation by setting:
    y=max(0,x) --> if x > 0 then y = x, otherwise y = 0
    
In distinguishing cats and dogs, conditional correlation helps to NN to learn
relevant features and patterns by sparse activation of neurons. For example:

If an image has pointed ears - neurons detecting this feature become active → 
leading to a "cat" classification.
If an image has floppy ears - different neurons detecting this feature get 
activated → leading to a "dog" classification.


d) Create a simple Python script that shows the initialization of a small neural 
    network with a hidden layer, applies the ReLU function and illustrates the 
    conditional correlation.  
'''

from numpy import load
np.warnings.filterwarnings('ignore', 'overflow')

photos = load('C:/Users/srivi/Documents/ML_data/dogs_vs_cats_photos2_56x56.npy')
labels = load('C:/Users/srivi/Documents/ML_data/dogs_vs_cats_labels2_56x56.npy')
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
    return np.maximum(x, 0)

def derivative_relu(x): 
    return x > 0

def softmax_activationfn(x):
    y = np.exp(x)/sum(np.exp(x))
    return y

def sigmoid_activationfn(y):
    sig = 1 / (1+e**(-y))
    return sig

def derivative_sigmoid_fn(y):
    dsig = np.exp(-y)/ (1+np.exp(-y))
    return dsig

def error(y, y_pred):
    err = y - y_pred
    return err

def forward_propagation(x,wi,wo,bi,bo):
    #print(x.shape,wi.shape)
    #print(((perceptron_npdot(x,wi))+bi).shape)
    hx = perceptron_npdot(x,wi)+bi
    ho  = relu_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = sigmoid_activationfn(yh)
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
    #print(X.shape)
    y_pred, h = forward_propagation(X,wi,wo,bi,bo)
    wi, wo = back_propagation(h, wi, wo, bi, bo, X, y, y_pred, lr)

#y_pred=y_pred.round(1)#.astype(int)
#print(y)
#print(y_pred)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y, y_pred)
print(accuracy)



















