# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 06:14:38 2025

@author: srivi
🔥 Train a Character-Based LSTM on Shakespeare Text (PyTorch)
Now, we'll train our LSTM model on real Shakespeare text instead of random sequences.

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset  # Hugging Face Datasets for Shakespeare data


# ✅ Step 2: Load & Preprocess Shakespeare Dataset

# Load the Shakespeare dataset (tiny version)
dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)  
text = dataset["train"]["text"]  # Get text data

# Remove special characters and normalize whitespace
text = " ".join(text).replace("\n", " ").strip()

# Define vocabulary (unique characters)
characters = sorted(set(text))  
vocab_size = len(characters)
char_to_idx = {c: i for i, c in enumerate(characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Convert text into indices
encoded_text = [char_to_idx[c] for c in text]

print(f"Total Characters: {len(text)}, Unique Characters: {vocab_size}")




# ✅ Step 3: Create Dataset & DataLoader
# We split text into sequences of 100 characters for training. 

class ShakespeareDataset(Dataset):
    def __init__(self, text_indices, seq_length=100):
        self.data = text_indices
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        X = torch.tensor(self.data[idx:idx + self.seq_length], dtype=torch.long)
        Y = torch.tensor(self.data[idx + 1:idx + self.seq_length + 1], dtype=torch.long)
        return X, Y

# Create dataset
seq_length = 100  
dataset = ShakespeareDataset(encoded_text, seq_length)

# DataLoader for batch processing
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")


# ✅ Step 4: Define LSTM Model in PyTorch

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256, num_layers=2):
        super(CharLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Embedding layer
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)  # LSTM
        self.fc = nn.Linear(hidden_size, vocab_size)  # Fully connected layer

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

# Create model instance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharLSTM(vocab_size).to(device)
print(model)

'''✅ LSTM model is ready for training!
Embedding → LSTM (256 units) → Fully Connected.
'''



#✅ Step 5: Train the Model

# Training hyperparameters
num_epochs = 50
learning_rate = 0.002

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    total_loss = 0
    hidden = None  # Reset hidden state at the beginning of each epoch

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        optimizer.zero_grad()
        output, hidden = model(X_batch, hidden)

        # Detach hidden states to prevent backprop through the entire sequence
        h_n, c_n = hidden
        h_n = h_n.detach()
        c_n = c_n.detach()
        hidden = (h_n, c_n)

        # Reshape for loss function
        loss = criterion(output.view(-1, vocab_size), Y_batch.view(-1))
        loss.backward(retain_graph=True)
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

'''✅ Model is now training!

Uses mini-batches (batch size = 64).
Resets hidden state at the start of each epoch.
Uses Adam optimizer + CrossEntropyLoss.
'''



#✅ Step 6: Generate Shakespeare-like Text


def generate_text(model, start_char="H", length=500):
    model.eval()
    char_idx = char_to_idx[start_char]  # Convert start character to index
    input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    hidden = None
    generated_text = start_char

    for _ in range(length):
        with torch.no_grad():
            output, hidden = model(input_seq, hidden)
            probs = torch.softmax(output[:, -1, :], dim=-1)  
            char_idx = torch.multinomial(probs, 1).item()
            generated_text += idx_to_char[char_idx]
            input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

    return generated_text

# Generate text
print(generate_text(model, start_char="T", length=500))



'''🎯 Summary
Step	Description
(1) Load Data	Used real Shakespeare text from Hugging Face.
(2) Preprocess Data	Converted characters to indices.
(3) Create Dataset	Used 100-char sequences for training.
(4) Define Model	LSTM (256 hidden units, 2 layers).
(5) Train Model	Used CrossEntropy + Adam Optimizer.
(6) Generate Text	Created Shakespeare-like text!
'''