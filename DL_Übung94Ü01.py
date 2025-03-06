# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 21:07:20 2025

@author: srivi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 05:51:37 2025
You have a neural network that is used to classify images into two categories: 
    “cats” and “non-cats”. The input images are 64x64 pixels in size and in 
    grayscale. Each image is converted into a 1D vector of size 4096 (64x64), 
    with each pixel value serving as an input value for the neural network. 
    The output layer of the network should indicate the probability that the 
    image shows a cat. 

a) Determine the dimensions of the weight matrix W and the bias vector b for 
    the first hidden layer if it contains 100 neurons. 

b) Choose a suitable activation function for the hidden layer from the standard
    activation functions (e.g. ReLU, Sigmoid, Tanh) and justify your choice. 

c) Implement a function forward_propagation in Python that performs forward 
    propagation for an input image. The function should return the activation 
    of the hidden layer and the output layer (before applying the activation 
                                              function). 

d) Explain how the softmax activation function is applied to the output layer 
    to obtain a probability and why this function is appropriate for the output 
    layer in this case.  
@author: srivi
"""
from os import listdir,makedirs
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
from math import e



from numpy import load
np.warnings.filterwarnings('ignore', 'overflow')

photos = load('C:/Users/srivi/Documents/ML_data/dogs_vs_cats_photos2_64x64.npy')
labels = load('C:/Users/srivi/Documents/ML_data/dogs_vs_cats_labels2_64x64.npy')
print(photos.shape, labels.shape)

X = photos.reshape(25000,4096)
y = labels.reshape(25000,1)

# flattend image contains 4096 pixels hence the ipLayerSize = 4096
# hdLayerSize = 4096
# its a binary problem : to classify as cats or non cats, so opLayerSize = 1
# weights = array(4096,100) and bias = array(100,1)

def initialise():
    ipLayerSize, hdLayerSize, opLayerSize = 4096,100,1
    wi = np.random.rand(ipLayerSize, hdLayerSize) 
    bi = np.zeros((1, hdLayerSize))
    wo = np.random.rand(hdLayerSize, opLayerSize) 
    bo = np.zeros((1, opLayerSize))
    print(wi.shape, wo.shape, bi.shape, bo.shape)
    return wi, wo, bi, bo


def perceptron_npdot(x, w): 
    y = np.dot(x, w)
    return y

# classification of cats amd non cats images are complex non linear problems
# images have lots of features like eyes, ears etc
# Relu is best here, as it applies conditional correlation, where cetrain neurons
# are activated when cats features are identified and certain others are activated
# when non cat features are identified. It allows neural networks to learn complex, 
# hierarchical patterns efficiently

def relu_activationfn(x):
    return np.maximum(x, 0)

def derivative_relu(x): 
    return x > 0

# Softmax is commonly used in the output layer. It converts all the outputs to 
# probabilities, with a total sum of probabilities of all output is always 1. 
# For example, if the network predicts:
# P(cat)=0.85,P(other)=0.15 - means the model is 85% confident the image is a cat.
# one output is significantly larger, softmax boosts its probability, enforcing a 
# clear decision.

def softmax_activationfn(x):
    y = np.exp(x)/sum(np.exp(x))
    return y


def error(y, y_pred):
    err = y - y_pred
    return err

# The function should return the activation of the hidden layer and the output 
# layer (before applying the activation function). 

def forward_propagation(x,wi,wo,bi,bo):
    #print(x.shape,wi.shape)
    #print(((perceptron_npdot(x,wi))+bi).shape)
    hx = perceptron_npdot(x,wi)+bi
    ho  = relu_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = softmax_activationfn(yh)
    return yh, ho


wi, wo, bi, bo = initialise()
lr = 0.3 
epochs = 1000

for epoch in range(epochs): 
    #print(X.shape)
    y_pred, h = forward_propagation(X,wi,wo,bi,bo)



















