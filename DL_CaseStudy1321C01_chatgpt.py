# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:15:52 2025

@author: srivi
"""

# -*- coding: utf-8 -*-
"""
Fixed version of Simple Character-based RNN for language modeling.
"""
import numpy as np
import random
import matplotlib.pyplot as plt

# ✅ Step 1: Create an artificial alphabet & generate a 1000-character sequence
alphabet = list("ABCDEFGHIJKLMNO")  # 15 unique characters
char_to_idx = {c: i for i, c in enumerate(alphabet)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Generate a sequence of 1000 random characters
sequence_length = 1000
char_sequence = [random.choice(alphabet) for _ in range(sequence_length)]
indexed_sequence = np.array([char_to_idx[c] for c in char_sequence])  # Convert to indices

# ✅ Step 2: Implement an RNN with an embedding layer & simple RNN cell
class SimpleRNN:
    def __init__(self, vocab_size=15, input_size=10, hidden_size=20, output_size=15):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.output_size = output_size

        # Initialize embedding layer
        self.embedding = np.random.randn(vocab_size, input_size) * 0.01

        # Initialize weight matrices
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # Hidden to output
        
        # Initialize biases
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs, h_prev):
        """
        Forward pass through RNN
        :param inputs: List of character indices
        :param h_prev: Previous hidden state
        :return: Final output, last hidden state, hidden states over time
        """
        h = h_prev
        hidden_states = []
        outputs = []
        
        for idx in inputs:
            x = self.embedding[idx].reshape(-1, 1)  # Convert index to embedding vector
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)  # Hidden state update
            y = np.dot(self.Why, h) + self.by  # Compute output
            
            hidden_states.append(h)
            outputs.append(y)
        
        return outputs, h, hidden_states  # Return outputs and last hidden state

# ✅ Step 3: Prepare training data using truncated BPTT
def create_batches(indexed_sequence, batch_size=32, seq_len=5):
    """
    Splits data into mini-batches for training.
    :return: List of (input sequence, target sequence) pairs
    """
    X_batches = []
    Y_batches = []
    
    num_batches = len(indexed_sequence) // (batch_size * seq_len)
    
    for i in range(num_batches):
        start_idx = i * batch_size * seq_len
        batch_X = []
        batch_Y = []
        
        for j in range(batch_size):
            start_seq = start_idx + j * seq_len
            batch_X.append(indexed_sequence[start_seq:start_seq + seq_len])
            batch_Y.append(indexed_sequence[start_seq + 1:start_seq + seq_len + 1])  # Next character
        
        X_batches.append(np.array(batch_X))
        Y_batches.append(np.array(batch_Y))
    
    return X_batches, Y_batches

# ✅ Step 4: Define Loss & SGD Optimization
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Stability trick
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    """
    Computes the cross-entropy loss.
    """
    probs = softmax(y_pred)  # Convert logits to probabilities
#    log_probs = -np.log(probs[y_true, np.arange(y_pred.shape[1])])
    batch_size = y_pred.shape[1]
    log_probs = -np.log(probs[y_true.flatten(), np.arange(batch_size * seq_len)])
    return np.mean(log_probs)

def sgd_update(model, dWx, dWh, dWy, dbh, dby, lr=0.01):
    """
    Updates model parameters using Stochastic Gradient Descent (SGD).
    """
    model.Wxh -= lr * dWx
    model.Whh -= lr * dWh
    model.Why -= lr * dWy
    model.bh -= lr * dbh
    model.by -= lr * dby

# ✅ Step 5: Train the RNN for 100 Iterations
rnn = SimpleRNN()
epochs = 100
learning_rate = 0.01
batch_size = 32
seq_len = 5
hidden_dim = rnn.hidden_size

X_batches, Y_batches = create_batches(indexed_sequence)

losses = []
for epoch in range(epochs):
    total_loss = 0
    h_prev = np.zeros((hidden_dim, 1))  # Reset hidden state at the start of each epoch

    for X_batch, Y_batch in zip(X_batches, Y_batches):
        dWx, dWh, dWy = np.zeros_like(rnn.Wxh), np.zeros_like(rnn.Whh), np.zeros_like(rnn.Why)
        dbh, dby = np.zeros_like(rnn.bh), np.zeros_like(rnn.by)
        
        batch_loss = 0
        for i in range(batch_size):
            inputs, targets = X_batch[i], Y_batch[i]
            
            # Forward pass
            outputs, h_prev, hidden_states = rnn.forward(inputs, h_prev)
            y_pred = np.hstack(outputs)
            batch_loss += cross_entropy_loss(y_pred, targets)
            hidden_states[-1] = hidden_states[-1].reshape(hidden_dim, seq_len)

            
            # Backpropagation through time (BPTT)
            probs = softmax(y_pred)
            one_hot_targets = np.eye(15)[targets].T
            dWy += np.dot((probs - one_hot_targets), hidden_states[-1].T)        
            dby += np.sum(softmax(y_pred) - np.eye(15)[targets].T, axis=1, keepdims=True)
        
        # Normalize gradients
        dWy /= batch_size
        dby /= batch_size
        
        # Perform SGD step
        sgd_update(rnn, dWx, dWh, dWy, dbh, dby, learning_rate)
        
        total_loss += batch_loss / batch_size
    
    losses.append(total_loss / len(X_batches))
    print(f"Epoch {epoch+1}/{epochs}, Loss: {losses[-1]:.4f}")

# ✅ Final Step: Plot Loss
plt.plot(losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss over Time")
plt.legend()
plt.show()
