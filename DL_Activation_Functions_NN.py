# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:11:30 2025

@author: srivi
"""


''' 
Activation Function - to introduce non-linearity in the NN - without Activation
funtion, NN outputs linear mapping. applied to neurons in a layer.
- the function must be continuous function, returns op for all ip in 
definite range
- the function should be monotonically growing and should not change direction
- goog function should be non-linear (or curved)- activation doesnot allow a weight
to affect how much a neuron correlates with other weights
- good function and its derivative should be able to be effectively calculated

neuron with an activation function - incoming signal should be able to increase
or decrease the correlation of neuron with all other incoming signals.

Choosing the right Activation Function
Now that we have seen so many activation  functions, we need some logic / heuristics 
to know which activation function should be used in which situation. 
Good or bad – there is no rule of thumb.

However depending upon the properties of the problem we might be able to make 
a better choice for easy and quicker convergence of the network.

Sigmoid functions and their combinations generally work better in the case of classifiers
Sigmoids and tanh functions are sometimes avoided due to the vanishing gradient problem
ReLU function is a general activation function and is used in most cases these days
If we encounter a case of dead neurons in our networks the leaky ReLU function is the best choice
Always keep in mind that ReLU function should only be used in the hidden layers

As a rule of thumb, you can begin with using ReLU function and then move over to other 
activation functions in case ReLU doesn’t provide with optimum results'''



import numpy as np

def binary_step(x):
    if x<0:
        return 0
    else:
        return 1
    

'''Binary Step : if the input to the activation function is greater than 
a threshold, then the neuron is activated, else it is deactivated, 
i.e. its output is not considered for the next hidden layer'''

'''IMPORTANT: the gradient of the function became zero'''

def linear_function(x):
    return 4*x


'''Linear Function: the gradient here does not become zero, but it is a constant 
which does not depend upon the input value x at all. This implies that the 
weights and biases will be updated during the backpropagation process but the 
updating factor would be the same. '''


def sigmoid_function(x):
    z = (1/(1 + np.exp(-x)))
    return z

'''Sigmoid Function: a smooth S-shaped function and is continuously differentiable.'''

def derivative_sigmoid_function(x):
    dz = sigmoid_function(x)*(1-sigmoid_function(x))
    return dz

'''the sigmoid function is not symmetric around zero. So output of all the 
neurons will be of the same sign - maps the input range and output range between 
0 and 1. - provides only positive correlation

This can be addressed by scaling the sigmoid 
function which is exactly what happens in the tanh function.'''

'''for values greater than 3 or less than -3, will have very small gradients. 
As the gradient value approaches zero, the network is not really learning.'''


def tanh_function(x):
    z = (2/(1 + np.exp(-2*x))) -1
    return z

''' tanh(x) = 2sigmoid(2x)-1  is same as  tanh(x) = 2/(1+e^(-2x)) -1 '''

'''the range of values is between -1 to 1 and gradient of the tanh function is 
steeper -- otherwise it is similar to the sigmoid function'''

'''tanh is preferred over the sigmoid function since it is zero centered and 
the gradients are not restricted to move in a certain direction
- also provides negative correlation
'''


def relu_function(x):
    if x<0:
        return 0
    else:
        return x

'''ReLU stands for Rectified Linear Unit. 
-used only in the hidden layers, as the output is limited to 0 and 1
The main advantage of using the ReLU 
function over other activation functions is that it does not activate all the 
neurons at the same time.
- negative side of the graph, the gradient/slope value is zero. 
- slope of positive function is always 1 
Due to this reason, during the 
backpropogation process, the weights and biases for some neurons are not updated. 
This can create dead neurons which never get activated. This is taken care of by 
the ‘Leaky’ ReLU function.'''

def derivative_relu(x): 
    if x<0:
        return 0
    else:
        return 1

'''is a special function that takes the output of the relu function and calculates 
its slope at this point
- returns 1 if the input is >= 0, otherwise 0.
'''
def leaky_relu_function(x):
    if x<0:
        return 0.01*x
    else:
        return x

''' Instead of defining the Relu function as 0 for negative values of x, we define 
it as an extremely small linear component of x. the gradient of the left side of 
the graph comes out to be a non zero value. Hence we would no longer encounter 
dead neurons in that region'''


def parameterised_relu_function(x):
    a = 0.01
    if x<0:
        return a*x
    else:
        return x

'''"a" is a trainable parameter. The network also learns the value of ‘a‘ for 
faster and more optimum convergence.'''

'''The derivative of the function would be same as the Leaky ReLu function, 
except the value 0.01 will be replcaed with the value of a.
f'(x) = 1, x>=0 
      = a, x<0 '''


def exponential_linearunit_function(x, a):
    if x<0:
        return a*(np.exp(x)-1)
    else:
        return x

'''Exponential Linear Unit or ELU for short is also a variant of Rectiufied 
Linear Unit (ReLU) that modifies the slope of the negative part of the function. 
Unlike the leaky relu and parametric ReLU functions, instead of a straight line, 
ELU uses a log curve for defning the negatice values'''

def swish_function(x):
    return x/(1-np.exp(-x))

'''Swish is as computationally efficient as ReLU and shows better performance 
than ReLU on deeper models.  The values for swish ranges from negative infinity
to infinity. The function is defined as –

f(x) = x*sigmoid(x)  is same as  f(x) = x/(1-e^-x)'''


def softmax_function(x):
    z = np.exp(x)
    z_ = z/z.sum()
    return z_

'''Softmax function 
- exposes the input values and divides then by the sum of the values of the 
layer
- often described as a combination of multiple sigmoids.
- sigmoid is widely used for binary classification problems. The softmax 
function can be used for multiclass classification problems. represent the 
probability for the data point belonging to each class. Note that the sum of all 
the values is 1.'''

- Alternative: Why Not Use Sigmoid Instead?
For binary classification (cat vs. dog), another option is the sigmoid function 
and applying binary cross-entropy loss. However, softmax is still preferred in 
multi-class cases, and even for binary cases where a one-hot encoded output is used.




