# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:45:18 2025

@author: srivi
"""

import numpy as np

class Tensor:
    def __init__(self, data):
        self.data  = np.array(data)#, dtype = np.float32)
        
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)
        
    def __repr__(self): #Return a string containing a printable representation of an object
        return f"Tensor({self.data})"

# Example usage:
t1 = Tensor([1, 2, 3])
t2 = Tensor([4, 5, 6])
t3 = t1 + t2  # Calls __add__
print(t3)  # Output: Tensor([5. 7. 9.])


class Tensor:
    def __init__(self, data, creators=None, creation_op=None):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None  # Store gradient (initialized as None)
        self.creators = creators  # Parent tensors that led to this tensor
        self.creation_op = creation_op  # Operation that created this tensor

    def __add__(self, other):
        if isinstance(other, Tensor):
            result = Tensor(self.data + other.data, creators=(self, other), creation_op="add")
            return result
        else:
            return Tensor(self.data + other)  # Allow scalar addition

    def backward(self, grad=None):
        """Computes gradients for autograd."""
        if grad is None:
            grad = np.ones_like(self.data)  # Default gradient of 1 if not specified
        
        self.grad = grad  # Store gradient

        # Propagate gradient to parent tensors if they exist
        if self.creation_op == "add":
            self.creators[0].backward(grad)  # Pass the gradient to the first tensor
            self.creators[1].backward(grad)  # Pass the gradient to the second tensor

    def __repr__(self):
        return f"Tensor({self.data}, grad={self.grad})"

# Example usage:
t1 = Tensor([1, 2, 3])
t2 = Tensor([4, 5, 6])
t3 = t1 + t2  # t3 stores t1 and t2 as its creators

print(t3)  # Output: Tensor([5. 7. 9.], grad=None)

# Compute gradients (calling backward on t3)
t3.backward()
print(t1.grad)  # Output: [1. 1. 1.] (Gradient of addition is 1 w.r.t each input)
print(t2.grad)  # Output: [1. 1. 1.]


   