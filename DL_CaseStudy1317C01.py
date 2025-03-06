# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:14:20 2025

@author: srivi

You work as a data scientist in a company that specializes in the development of deep learning 
specialized in the development of deep learning models. Your task is to create a simple 
neural network that is capable of learning XOR operations. 
You will use the concepts of autograd and stochastic gradient descent 
(SGD) to train the network. Proceed as follows: 

a) Define a tensor class that supports basic operations such as matrix multiplication 
    and addition, as well as the ability for automatic gradient descent (autograd).
    (autograd). The class should also contain a method backward(), 
    which propagates the gradient backwards through the network. 

b) Create a class SGD that acts as an optimizer. This class should contain methods
    zero() and step() to zero the gradients and perform the weight updates.
    updates. 

c) Implement the Linear and Sequential layer types. The Linear class should 
    initialize weights and bias and have a forward() method. The 
    Sequential class should manage a list of layers and coordinate their execution 
    in the forward in the forward() method.    
    
d) Create a neural network model with the sequential class that consists of two linear 
    two linear layers to learn the XOR function. The first layer 
    should project the input to a higher dimension (e.g. from 2 to 3 neurons),
    and the second layer should reduce it to an output value. 

e) Use the SGD optimizer to train your model. Perform the training 
    for 10 epochs and output the loss after each epoch. 
    
"""
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._prev = set()

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad})"

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = backward
        out._prev = {self, other}
        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)

        def backward():
            if self.requires_grad:
                self.grad += out.grad
            if other.requires_grad:
                other.grad += out.grad

        out._backward = backward
        out._prev = {self, other}
        return out

    def backward(self):
        if not self.requires_grad:
            return

        # Topological ordering
        topo_order = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited:
                visited.add(tensor)
                for prev in tensor._prev:
                    build_graph(prev)
                topo_order.append(tensor)

        build_graph(self)

        # Initialize gradient
        self.grad = np.ones_like(self.data)

        for tensor in reversed(topo_order):
            tensor._backward()

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(out_features, in_features) * 0.1, requires_grad=True)
        self.b = Tensor(np.zeros((out_features, 1)), requires_grad=True)

    def forward(self, x):
        return self.W @ x + self.b

class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

class SGD:
    def __init__(self, parameters, lr=0.1):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        """Gradient descent update"""
        for param in self.parameters:
            if param.requires_grad:
                param.data -= self.lr * param.grad
                param.grad.fill(0)

model = Sequential(
    Linear(2, 3),
    Linear(3, 1)
)

X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y_train = np.array([[0, 1, 1, 0]])

optimizer = SGD(parameters=[model.layers[0].W, model.layers[0].b,
                            model.layers[1].W, model.layers[1].b], lr=0.1)

epochs = 5000
for epoch in range(epochs):
    total_loss = 0

    for i in range(X_train.shape[1]):
        x = Tensor(X_train[:, i].reshape(2, 1), requires_grad=True)
        y = Tensor(Y_train[:, i].reshape(1, 1), requires_grad=True)

        output = model.forward(x)

        loss = Tensor(np.mean((output.data - y.data) ** 2), requires_grad=True)
        total_loss += loss.data

        loss.backward()
        optimizer.step()

    if epoch % 500 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.6f}")

print("\n🔍 XOR Predictions:")
for i in range(X_train.shape[1]):
    x = Tensor(X_train[:, i].reshape(2, 1))
    y_pred = model.forward(x)
    prediction = 1 if y_pred.data >= 0.5 else 0
    print(f"Input: {X_train[:, i]}, Prediction: {prediction}")
  
    
    