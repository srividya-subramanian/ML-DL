# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:01:17 2025

@author: srivi

Here’s the real PyTorch implementation of a character-based LSTM for language modeling. 🚀

📌 Plan
Load & preprocess data (A-Z sequences).
Define the PyTorch LSTM model (Embedding → LSTM → Output).
Train the model (Cross-Entropy Loss + Adam Optimizer).
Generate text from the trained model (sample predictions).

"""
# -----✅ Step 2: Load & Prepare Data -----
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random

# Load dataset (random A-Z sequences)
num_sequences = 1000
seq_length = 50
characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
char_to_idx = {c: i for i, c in enumerate(characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Generate synthetic dataset
data = ["".join(random.choices(characters, k=seq_length)) for _ in range(num_sequences)]

# Convert characters to indices
X_data = [[char_to_idx[c] for c in seq[:-1]] for seq in data]  # Inputs
Y_data = [[char_to_idx[c] for c in seq[1:]] for seq in data]  # Targets

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_data, dtype=torch.long)  # Shape: (1000, 49) # Inputs
Y_tensor = torch.tensor(Y_data, dtype=torch.long)  # Shape: (1000, 49) # Targets

print(f"Data shape: {X_tensor.shape}, {Y_tensor.shape}")

'''🔍 Explanation
Maps characters (A-Z) to indices (0-25).
Creates input sequences (X) & target sequences (Y) (shifted by 1).
Converts data to PyTorch tensors (for training).
'''


# -----✅ Step 3: Define LSTM Model in PyTorch -----

class CharLSTM(nn.Module):
    def __init__(self, vocab_size=26, embed_size=128, hidden_size=256, num_layers=1):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM layer
        self.fc = nn.Linear(hidden_size, vocab_size)  # Fully connected Output layer

    def forward(self, x, h=None):
        x = self.embedding(x)  # Convert indices to embeddings
        out, h = self.lstm(x, h)  # Pass through LSTM
        out = self.fc(out)  # Fully connected layer
        return out, h

# Create model instance
model = CharLSTM()

print(model)

'''🔍 Explanation
Layer	Function
Embedding Layer	Maps input characters to a 128-dim vector.
LSTM Layer	Processes sequence with 256 hidden units.
Fully Connected Layer	Predicts next character probabilities.
'''

# -----Step 4: Define Training Parameters -----

# Hyperparameters
batch_size = 32
num_epochs = 50
learning_rate = 0.001
seq_length = 49  # Input length (one character less)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()  # Softmax + Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Move model to GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Move data to device
X_tensor = X_tensor.to(device)
Y_tensor = Y_tensor.to(device)

'''🔍 Explanation
Uses CrossEntropyLoss (combines Softmax + Log Loss).
Uses Adam Optimizer for better convergence.
Moves model & data to GPU (if available).
'''

# ----- Step 5: Train the Model -----

from torch.utils.data import TensorDataset, DataLoader

# Create DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    
    for X_batch, Y_batch in dataloader:
        optimizer.zero_grad()  # Reset gradients
        output, _ = model(X_batch)  # Forward pass

        # Reshape output & target to match (batch*seq_len, vocab_size)
        loss = criterion(output.view(-1, 26), Y_batch.view(-1))

        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

'''🔍 Explanation
Uses mini-batches (batch size = 32) for efficient training.
Computes loss over entire sequence (49 characters).
Performs Backpropagation & Gradient Descent.
'''


# ----- Step 6: Generate Text from Trained Model -----

def generate_text(model, start_char="A", length=100):
    model.eval()  # Set to evaluation mode
    
    char_idx = char_to_idx[start_char]  # Start character
    input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)  # Convert to tensor
    
    hidden = None
    generated_text = start_char
    
    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)  # Forward pass
            probs = torch.softmax(output[:, -1, :], dim=-1)  # Get last char probabilities
            
            char_idx = torch.multinomial(probs, 1).item()  # Sample next char
            generated_text += idx_to_char[char_idx]
            
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)  # Update input
    
    return generated_text

# Generate new text
print(generate_text(model, start_char="H", length=200))


'''
🔍 Explanation
Starts with a given character (A).
Predicts next character using LSTM hidden state.
Generates a sequence of 100 characters.

🎯 Summary
Step	             Description
(1) Load Data	     Created random sequences of A-Z and converted to PyTorch tensors.
(2) Define Model	 Built an LSTM model (Embedding → LSTM → FC).
(3) Train Model	     Used mini-batch training with Adam optimizer.
(4) Generate Text	 Sampled new character sequences from the trained model.


'''























