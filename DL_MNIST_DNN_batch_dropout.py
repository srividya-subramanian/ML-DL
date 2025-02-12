# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 08:56:11 2025

@author: srivi
"""

import numpy as np 
import sys, numpy as np
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
images, labels = (x_train[0:1000].reshape(1000,28*28)/255, y_train[0:1000])

'''training sample'''
one_hot_labels = np.zeros((len(labels),10))

for i,l in enumerate(labels):
    one_hot_labels[i][l] = 1
labels = one_hot_labels


'''testing sample'''
test_images = x_test.reshape(len(x_test),28*28) / 255 
test_labels = np.zeros((len(y_test),10))

for i,l in enumerate(y_test):
    test_labels[i][l] = 1


np.random.seed(1)
def perceptron_npdot(x, w): 
    return np.dot(x, w)
def relu(x):
    return (x >= 0) * x 
def relu2deriv(output): 
    return output >= 0 
def error(y, y_pred):
    return (y - y_pred)** 2
def batch_delta(y, y_pred, batch_size):
    return (y - y_pred)/batch_size

'''Initialising parameters: lr, w, batch size, epochs'''
batch_size = 100
alpha, iterations = (0.001, 300)

pixels_per_image, hidden_size, num_labels = (784, 100, 10)
weights_0_1 = 0.2 * np.random.random((pixels_per_image,hidden_size)) - 0.1
weights_1_2 = 0.2 * np.random.random((hidden_size,num_labels)) - 0.1


for j in range(iterations):
    error, correct_cnt = (0.0, 0)

    for i in range(int(len(images) / batch_size)):
        batch_start, batch_end = ((i * batch_size),((i+1)*batch_size))
        
        layer_0 = images[batch_start:batch_end]
        labels_0 = labels[batch_start:batch_end]
        
        layer_1 = perceptron_npdot(layer_0,weights_0_1)
        layer_1 = relu(layer_1)
       
        dropout_mask = np.random.randint(2,size=layer_1.shape)
        layer_1 *= dropout_mask * 2
        layer_2 = perceptron_npdot(layer_1,weights_1_2)
        
#        error += error(labels_0, layer_2.round(2))
        
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer_2[k:k+1]) ==
                               np.argmax(labels[batch_start+k:batch_start+k+1]))
            
            layer_2_delta = batch_delta(labels_0, layer_2, batch_size)
            layer_1_delta = perceptron_npdot(layer_2_delta, weights_1_2.T) * relu2deriv(layer_1)
            layer_1_delta *= dropout_mask
            
            weights_1_2 += alpha * perceptron_npdot(layer_1.T, layer_2_delta)
            weights_0_1 += alpha * perceptron_npdot(layer_0.T, layer_1_delta)
    
    if(j%10 == 0): 
        test_error = 0.0
        test_correct_cnt = 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i+1]
        
            layer_1 = perceptron_npdot(layer_0,weights_0_1)
            layer_1 = relu(layer_1)

            layer_2 = perceptron_npdot(layer_1, weights_1_2)

        #sys.stdout.write(" Test-Error:" + str(error/float(len(test_images)))[0:5] +
                         #" Test-KKR:" + str(correct_cnt/float(len(test_images))))


