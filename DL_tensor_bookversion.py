# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 06:19:53 2025

@author: srivi
"""

import numpy as np
class Tensor (object):
    def __init__(self, data):
        self.data = np.array(data)
    def __add__(self, other):
        return Tensor(self.data + other.data)
    def __repr__(self):
        return str(self.data.__repr__())
    def __str__(self):
        return str(self.data.__str__())

x = Tensor([1,2,3,4,5])
print(x)
y = x + x
print(y)



import numpy as np
class Tensor (object):
    def __init__(self, data, creators=None, creation_op=None):
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
    def backward(self, grad):
        self.grad = grad
        if(self.creation_op == "add"):
            self.creators[0].backward(grad)
            self.creators[1].backward(grad)
    def __add__(self, other):
        return Tensor(self.data + other.data,
                      creators=[self,other],
                      creation_op="add")
    def __repr__(self):
        return str(self.data.__repr__())
    def __str__(self):
        return str(self.data.__str__())
    
    
x = Tensor([1,2,3,4,5])
y = Tensor([2,2,2,2,2])
z=x+y
z.backward(Tensor(np.array([1,1,1,1,1])))