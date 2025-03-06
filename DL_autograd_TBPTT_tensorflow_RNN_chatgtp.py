# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 13:01:42 2025
🚀 Implementing Truncated Backpropagation Through Time (TBPTT) in TensorFlow
Since Backpropagation Through Time (BPTT) can be computationally expensive for 
long sequences, we use Truncated BPTT (TBPTT) to limit gradient updates to a fixed 
number of time steps rather than propagating through the entire sequence.

🔹 How TBPTT Works
Instead of unrolling the entire sequence, we break it into smaller chunks 
(e.g., 10-20 time steps).
Backpropagation occurs only within each chunk, reducing memory usage and 
improving efficiency.
Training still captures long-term dependencies through overlapping truncated sequences.

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
time_steps = 50  # Total sequence length
input_dim = 1
truncated_steps = 10  # Number of time steps per truncated segment
epochs = 100

# Create dataset
X_train = generate_time_series(batch_size, time_steps)
y_train = generate_time_series(batch_size, time_steps)  # Target

# Define a simple RNN model
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', return_sequences=True, input_shape=(None, input_dim)),
    tf.keras.layers.Dense(1)
])

# Loss function & optimizer
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

# Training loop with TBPTT
loss_history = []

for epoch in range(epochs):
    total_loss = 0.0
    num_truncated_batches = time_steps // truncated_steps  # Number of chunks per sequence

    for i in range(num_truncated_batches):
        # Extract truncated sequence
        start_idx = i * truncated_steps
        end_idx = start_idx + truncated_steps
        X_truncated = X_train[:, start_idx:end_idx, :]
        y_truncated = y_train[:, start_idx:end_idx, :]

        with tf.GradientTape() as tape:
            y_pred = model(X_truncated)  # Forward pass on truncated sequence
            loss = loss_fn(y_truncated, y_pred)  # Compute loss

        # Compute gradients (only for truncated steps)
        gradients = tape.gradient(loss, model.trainable_variables)
        
        # Update model weights
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss.numpy()  # Accumulate loss

    avg_loss = total_loss / num_truncated_batches
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

# Plot loss curve
plt.plot(loss_history, label="Training Loss (TBPTT)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()


'''
🔹 Key Changes for TBPTT
Truncate the sequence: Instead of backpropagating through all time steps, we divide 
it into smaller chunks of truncated_steps (e.g., 10 time steps).

Perform BPTT separately on each chunk:
The loop over num_truncated_batches processes small segments at a time.
Each truncated segment is fed to the model independently.
The gradients are computed and updated for each segment separately.

Memory-efficient: Instead of keeping track of the entire sequence, we only store 
gradients for small chunks, reducing memory usage.

'''