# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:24:16 2025

@author: srivi
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic time-series data
def generate_time_series(batch_size, time_steps):
    freq = np.random.uniform(0.1, 1.0, (batch_size, 1))  # Random frequency
    phase = np.random.uniform(0, 2 * np.pi, (batch_size, 1))  # Random phase
    time = np.linspace(0, 10, time_steps)  # Time steps
    data = np.sin(freq * time + phase)  # Sinusoidal pattern
    return np.expand_dims(data, axis=-1)  # Shape: (batch_size, time_steps, 1)

# Hyperparameters
batch_size = 32
time_steps = 50
input_dim = 1
epochs = 100

# Create dataset
X_train = generate_time_series(batch_size, time_steps)
y_train = generate_time_series(batch_size, time_steps)  # Target (can be shifted for forecasting)

# Define a simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(None, input_dim)),
    tf.keras.layers.Dense(1)
])

# Loss function & optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Training loop with BPTT
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(X_train)  # Forward pass
        loss = loss_fn(y_train, y_pred)  # Compute loss

    # Compute gradients using BPTT
    gradients = tape.gradient(loss, model.trainable_variables)

    # Update model weights
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Track loss
    loss_history.append(loss.numpy())
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.numpy():.4f}")

# Plot loss curve
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

