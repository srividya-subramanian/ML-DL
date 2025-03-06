# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:47:54 2025

@author: srivi

🔹 Automatic Gradient Computation (Autograd) in Deep Learning
Autograd is a technique used in deep learning frameworks to automatically compute 
gradients during backpropagation. Instead of manually deriving gradients, autograd 
tracks operations and calculates derivatives efficiently.

🔹 How Autograd Works
Tracks Operations – It records mathematical operations applied to tensors.
Computes Gradients – Uses chain rule to compute derivatives automatically.
Optimizes Training – Helps update model parameters in gradient-based optimization.

🔹 Why Use Autograd?
✅ Saves Time – No need to compute derivatives manually.
✅ Handles Complex Models – Supports deep neural networks efficiently.
✅ Works with Any Function – Computes gradients dynamically.


In TensorFlow, automatic differentiation is handled using tf.GradientTape.
 It records operations applied to tensors and computes gradients automatically 
 during backpropagation.








"""
import tensorflow as tf

# Define a variable x with gradient tracking
x = tf.Variable(2.0)

# Use GradientTape to compute the gradient
with tf.GradientTape() as tape:
    y = x**2 + 3*x + 5  # Function: f(x) = x^2 + 3x + 5

'''Why Use with for GradientTape()?
When computing gradients, TensorFlow needs to "record" operations. tf.GradientTape() 
only records operations inside the with block.
Outside the block, TensorFlow does not track computations, so gradients cannot be 
computed correctly.
'''
# Compute dy/dx
grad = tape.gradient(y, x)
print("Gradient at x=2:", grad.numpy())  # Expected output: 7 (2x + 3 = 2(2) + 3)



#🔹 Example: Autograd in a Neural Network

# Define a simple model
w = tf.Variable(3.0)  # Weight parameter
b = tf.Variable(2.0)  # Bias parameter

# Define a loss function: L = (w * x + b - y)^2
def loss_fn(x, y):
    return (w * x + b - y) ** 2 # MSE

# Sample data
x_train, y_train = 4.0, 10.0  # Example input-output pair

# Compute gradients
with tf.GradientTape() as tape:
    loss = loss_fn(x_train, y_train)

# Get gradients
grad_w, grad_b = tape.gradient(loss, [w, b])
print("Gradient for w:", grad_w.numpy())
print("Gradient for b:", grad_b.numpy())

'''
🔹 Why Use tf.GradientTape?
✅ Tracks Operations Dynamically – Records forward computations automatically.
✅ Handles Neural Network Training – Computes gradients for optimization.
✅ Efficient Memory Management – Supports multiple gradient computations.

'''

