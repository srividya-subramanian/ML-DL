# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 05:27:31 2025

@author: srivi
"""

'''
Delta
- the purpose is to tell the previous layers to increase or decrease the input
- modifies the delta passes in Back propagation
- to generate layer_1_delta in back propagation: multiply back propagated
(layer_2_delta * weights) by the slope of the relu function at the point 
predicted in forward propagation (relu2deriv(layer_1))
- real goal is to adjust the weight in order to reduce the error -this step 
makes it more convenient for the network to leave the weights unchanged if the 
adjustments have little or no effect.

layer_1_delta - specifies how much larger or smaller the first hidden node of 
layer_1 should be in order to reduce the error
if there is no non-linearity: = np.dot(layer_2_delta, weights_1_2.T)



Function Forward Propagation Backpropagation delta

relu 
ones_and_zeros = (input > 0)        mask = output > 0
output = input*ones_and_zeros       deriv = output * mask


sigmoid 
output = 1/(1 + np.exp(-input))     deriv = output * (1-output)

Tanh 
output = np.tanh(input)             deriv = 1 – (output**2)

Softmax 
temp = np.exp(input)                temp = (output – true)
output/= np.sum(temp)               output = temp / len(true)
















'''
















