# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 06:20:39 2025

@author: srivi
"""

import numpy as np

class Tensor:
    def __init__(self, value, requires_grad=False):
        self.value = np.array(value, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.value) if requires_grad else None
        self._backward = lambda: None  # Function to compute gradients
        self._prev = set()  # Track previous operations

    def __repr__(self):
        return f"Tensor(value={self.value}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value + other.value, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += 1 * out.grad  # d(out)/d(self) = 1
            if other.requires_grad:
                other.grad += 1 * out.grad  # d(out)/d(other) = 1
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.value * other.value, requires_grad=self.requires_grad or other.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad += other.value * out.grad  # d(out)/d(self) = other
            if other.requires_grad:
                other.grad += self.value * out.grad  # d(out)/d(other) = self
        
        out._backward = _backward
        out._prev = {self, other}
        return out

    def backward(self):
        """Computes gradients using reverse-mode autograd."""
        topo_order = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for prev in tensor._prev:
                    build_graph(prev)
                topo_order.append(tensor)

        build_graph(self)
        self.grad = np.ones_like(self.value)  # Seed gradient (dL/dL = 1)

        for tensor in reversed(topo_order):
            tensor._backward()

# Example Usage
x = Tensor(2.0, requires_grad=True)
y = Tensor(3.0, requires_grad=True)

z = x * x + y * y + x * y  # Function: f(x, y) = x² + y² + xy
z.backward()  # Compute gradients

print("Value of z:", z.value)
print("Gradient of z w.r.t x:", x.grad)  # Expected: 2x + y = 2(2) + 3 = 7
print("Gradient of z w.r.t y:", y.grad)  # Expected: 2y + x = 2(3) + 2 = 8
