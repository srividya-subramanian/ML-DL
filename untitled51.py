# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:28:22 2025

@author: srivi
"""

import numpy as np

# ✅ (a) Implementing the Embedding Layer
class Embedding:
    def __init__(self, vocab_size, embedding_dim):
        """
        Initializes the embedding layer with a random weight matrix.
        :param vocab_size: Number of unique words in vocabulary
        :param embedding_dim: Size of each word vector
        """
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1  # Random initialization

    def forward(self, indices):
        """Retrieve embeddings for the given word indices."""
        return self.weights[indices]  # Lookup table


# ✅ (b) Implementing the RNN Cell with BPTT
class RNNCell:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes an RNN cell with three weight matrices.
        :param input_dim: Size of input vectors (embedding size)
        :param hidden_dim: Size of hidden state
        :param output_dim: Size of output (vocab size)
        """
        self.hidden_dim = hidden_dim

        # Weight matrices
        self.Wx = np.random.randn(hidden_dim, input_dim) * 0.1
        self.Wh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.Wy = np.random.randn(output_dim, hidden_dim) * 0.1

        # Bias terms
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))

    def forward(self, x, h_prev):
        """Computes the next hidden state and output."""
        h_next = np.tanh(self.Wx @ x + self.Wh @ h_prev + self.bh)  # Hidden state update
        y = self.Wy @ h_next + self.by  # Output computation
        return h_next, y


# ✅ (c) Implementing Cross-Entropy Loss
class CrossEntropyLoss:
    def forward(self, predictions, targets):
        """
        Computes cross-entropy loss.
        :param predictions: Logits (raw outputs) from the model
        :param targets: True class indices
        """
        targets = np.array(targets, dtype=np.int32).flatten()  # Ensure 1D array
        
        exp_preds = np.exp(predictions - np.max(predictions, axis=0, keepdims=True))  # Softmax
        probs = exp_preds / np.sum(exp_preds, axis=0, keepdims=True)  # Normalize probabilities
        
        # Ensure indexing is correct
        assert probs.shape[1] == targets.shape[0], f"Shape mismatch: probs {probs.shape}, targets {targets.shape}"
        
        loss = -np.log(probs[targets, np.arange(probs.shape[1])])  # Cross-entropy
        return np.mean(loss), probs

    def backward(self, probs, targets):
        """
        Computes the gradient of the loss w.r.t. the predictions.
        :param probs: Softmax probabilities
        :param targets: True class indices
        """
        probs[targets, np.arange(targets.shape[0])] -= 1  # Gradient of cross-entropy
        return probs / targets.shape[0]  # Normalize by batch size


# ✅ (d) Training Routine with BPTT
class RNNModel:
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, lr=0.1):
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.rnn_cell = RNNCell(embedding_dim, hidden_dim, output_dim)
        self.loss_fn = CrossEntropyLoss()
        self.lr = lr  # Learning rate

    def train(self, sequences, targets, epochs=500):
        """Trains the RNN on given sequences using Backpropagation Through Time (BPTT)."""
        for epoch in range(epochs):
            total_loss = 0

            # Initialize weight gradients
            dWx = np.zeros_like(self.rnn_cell.Wx)
            dWh = np.zeros_like(self.rnn_cell.Wh)
            dWy = np.zeros_like(self.rnn_cell.Wy)
            dbh = np.zeros_like(self.rnn_cell.bh)
            dby = np.zeros_like(self.rnn_cell.by)

            # Initialize hidden state
            h_prev = np.zeros((self.rnn_cell.hidden_dim, 1))

            # Store intermediate values for BPTT
            hs = [h_prev]
            xs = []
            ys = []
            targets_list = []

            # Forward pass for the whole sequence
            for i in range(len(sequences)):
                x = self.embedding.forward(sequences[i]).reshape(-1, 1)  # Lookup embedding
                h_next, y_pred = self.rnn_cell.forward(x, h_prev)  # Forward pass

                loss, probs = self.loss_fn.forward(y_pred.data, targets[i])
                total_loss += loss

                # Store values for BPTT
                xs.append(x)
                hs.append(h_next)
                ys.append(probs)
                targets_list.append(targets[i])

                h_prev = h_next  # Update hidden state

            # Backpropagation Through Time (BPTT)
            dh_next = np.zeros_like(h_prev)
            for i in reversed(range(len(sequences))):
                grad_y = self.loss_fn.backward(ys[i], np.array([targets_list[i]]))

                # Gradients for output layer
                dWy += grad_y @ hs[i+1].T
                dby += np.sum(grad_y, axis=1, keepdims=True)

                # Gradients for hidden state
                dh = self.rnn_cell.Wy.T @ grad_y + dh_next
                dh_raw = (1 - hs[i+1] ** 2) * dh  # tanh derivative

                # Gradients for RNN weights
                dbh += dh_raw
                dWx += dh_raw @ xs[i].T
                dWh += dh_raw @ hs[i].T

                # Pass the gradient back
                dh_next = self.rnn_cell.Wh.T @ dh_raw

            # Update weights using SGD
            self.rnn_cell.Wx -= self.lr * dWx
            self.rnn_cell.Wh -= self.lr * dWh
            self.rnn_cell.Wy -= self.lr * dWy
            self.rnn_cell.bh -= self.lr * dbh
            self.rnn_cell.by -= self.lr * dby

            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")


# ✅ Simple Dataset
vocab_size = 5  # Vocabulary of 5 words (IDs: 0-4)
embedding_dim = 4  # Each word maps to a 4D vector
hidden_dim = 6  # RNN hidden state size
output_dim = vocab_size  # Output size same as vocab size

# Toy sequences: Each word ID should predict the next word ID
sequences = np.array([0, 1, 2, 3])  # Word IDs in a sequence
targets = np.array([1, 2, 3, 4])  # Next word IDs

# ✅ Training
model = RNNModel(vocab_size, embedding_dim, hidden_dim, output_dim)
model.train(sequences, targets, epochs=500)
