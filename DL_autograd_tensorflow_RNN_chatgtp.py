# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:22 2025

@author: srivi

Autograd - automated differenciation system  - calculates gradients during training
    - tracks all operations such as matrix multiplications, activations etc, performed
        in forward pass
    - in the back pass, calculates gradients of loss function wrt parameters like 
        weights and baises by propagating the error backwords. Backpropagation is the 
        specific algorithm for updating weights in a neural network
    - uses tf.GradientTape() API
    - autograd is an automated calculation of gradients and backpropagation uses the
        gradients computed to update the model parameters during training

Training RNNs with TensorFlow and Autograd:

    Define the RNN architecture
    Forward Pass
    Compute Gradients
    update weights via backpropagation
    

"""

import tensorflow as tf

# Define a simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(None, 10)), 
    # 10 features as input
    tf.keras.layers.Dense(1)
])

# Define a loss function and an optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Dummy data (batch size of 32, 5 time steps, 10 features)
X_train = tf.random.normal((32, 5, 10))
y_train = tf.random.normal((32, 1))

# Training loop
for epoch in range(100):
    with tf.GradientTape() as tape:
       
        # Forward pass
        y_pred = model(X_train)
        
        # Calculate loss
        loss = loss_fn(y_train, y_pred)
    
    # Compute gradients using autograd
    gradients = tape.gradient(loss, model.trainable_variables)
    
    # Update model weights using optimizer
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    print(f"Epoch {epoch+1}: Loss = {loss.numpy()}")

