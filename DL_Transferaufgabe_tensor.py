# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:01:21 2025

@author: srivi
"""

import numpy as np 

class Tensor(object): 
    def __init__(self, data, creators=None, creation_op=None): 
        self.data = np.array(data) 
        self.creators = creators 
        self.creation_op = creation_op 
        self.grad = None 
        self.children = {} 
        if creators is not None: 
            for c in creators: 
                if self not in c.children: 
                    c.children[self] = 1 
                else: 
                    c.children[self] += 1 
 
    def all_children_grads_accounted_for(self): 
        for _, cnt in self.children.items(): 
            if cnt != 0: 
                return False 
        return True 
 
    def backward(self, grad=None, grad_origin=None): 
        if grad is None: 
            grad = Tensor(np.ones_like(self.data)) 
 
        if grad_origin is not None: 
            if self.children[grad_origin] == 0: 
                raise Exception("Cannot backprop more than once") 
            else: 
                self.children[grad_origin] -= 1 
 
        if self.grad is None: 
            self.grad = grad 
        else: 
            self.grad += grad 
 
        if self.creators is not None and (self.all_children_grads_accounted_for() or grad_origin is None): 
            if self.creation_op == "add": 
                self.creators[0].backward(self.grad, self) 
                self.creators[1].backward(self.grad, self) 
            elif self.creation_op == "neg": 
                self.creators[0].backward(self.grad.__neg__()) 
 
    def __add__(self, other): 
        if self.data.shape != other.data.shape: 
            raise ValueError("Tensors must be of the same shape") 
        return Tensor(self.data + other.data, creators=[self, other], creation_op="add") 
 
    def __neg__(self): 
        return Tensor(self.data * -1, creators=[self], creation_op="neg") 
 
    def __repr__(self): 
        return str(self.data.__repr__()) 
 
    def __str__(self): 
        return str(self.data.__str__()) 
 










# Beispiel für die Verwendung des Frameworks 

x = Tensor([1, 2, 3, 4, 5]) 
y = Tensor([2, 2, 2, 2, 2]) 
 
z = x + y 
z.backward(Tensor(np.array([1, 1, 1, 1, 1]))) 
print(x.grad)  # Sollte ein Tensor mit [1, 1, 1, 1, 1] sein 
print(y.grad)  # Sollte ein Tensor mit [1, 1, 1, 1, 1] sein 
 
a = Tensor([1, 2, 3, 4, 5]) 
b = -a 
b.backward(Tensor(np.array([1, 1, 1, 1, 1]))) 
print(a.grad)  # Sollte ein Tensor mit [-1, -1, -1, -1, -1] sein 


