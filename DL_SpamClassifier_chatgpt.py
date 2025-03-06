# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 05:09:25 2025

@author: srivi
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 04:36:15 2025

@author: srivi
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import codecs
import sys

# ✅ Read Spam Data
with codecs.open('C:/Users/srivi/Documents/velptec_K4.0016_2.0_DL_Buchmaterial/spam.txt',
                 "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

vocab, spam, ham = set(["<unk>"]), [], []

for row in raw:
    cleaned_row = row.strip()
    if cleaned_row:
        words = set(cleaned_row.split())
        spam.append(words)
        vocab.update(words)

# ✅ Read Ham Data
with codecs.open('C:/Users/srivi/Documents/velptec_K4.0016_2.0_DL_Buchmaterial/ham.txt',
                 "r", encoding='utf-8', errors='ignore') as f:
    raw = f.readlines()

for row in raw:
    cleaned_row = row.strip()
    if cleaned_row:
        words = set(cleaned_row.split())
        ham.append(words)
        vocab.update(words)

# ✅ Create Vocabulary Mapping
vocab = list(vocab)
w2i = {word: i for i, word in enumerate(vocab)}

# ✅ Convert Text Data to Indices
def to_indices(input, l=500):
    indices = []
    for line in input:
        # Truncate if longer than `l`
        line = list(line)[:l]  
        # Pad if shorter than `l`
        line += ["<unk>"] * (l - len(line))  
        idxs = [w2i[word] for word in line]
        indices.append(idxs)
    return indices    

spam_idx = to_indices(spam) 
ham_idx = to_indices(ham) 

# ✅ Split into Train/Test Sets
train_spam_idx = spam_idx[:-1000] 
train_ham_idx = ham_idx[:-1000] 
test_spam_idx = spam_idx[-1000:] 
test_ham_idx = ham_idx[-1000:] 

train_data, train_target = [], []
test_data, test_target = [], []

for i in range(max(len(train_spam_idx), len(train_ham_idx))): 
    train_data.append(train_spam_idx[i % len(train_spam_idx)]) 
    train_target.append([1]) 
    train_data.append(train_ham_idx[i % len(train_ham_idx)]) 
    train_target.append([0])

for i in range(max(len(test_spam_idx), len(test_ham_idx))): 
    test_data.append(test_spam_idx[i % len(test_spam_idx)]) 
    test_target.append([1]) 
    test_data.append(test_ham_idx[i % len(test_ham_idx)]) 
    test_target.append([0])

# ✅ Define Model
class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SpamClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x).sum(1)  # Sum along sequence length
        return self.fc(embedded)  # No activation (we use BCEWithLogitsLoss)

# ✅ Define Training Function
def train(model, input_data, target_data, batch_size=500, epochs=5, lr=0.01):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        iter_loss = 0
        n_batches = len(input_data) // batch_size
        for b_i in range(n_batches):
            batch_input = torch.tensor(input_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.long)
            batch_target = torch.tensor(target_data[b_i*batch_size:(b_i+1)*batch_size], dtype=torch.float)

            optimizer.zero_grad()
            predictions = model(batch_input)#.squeeze(1)  # Remove extra dimension
            loss = criterion(predictions, batch_target)
            loss.backward()
            optimizer.step()

            iter_loss += loss.item() / batch_size
            sys.stdout.write(f"\r\tLoss: {iter_loss / (b_i + 1):.4f}")

        print()
    return model

# ✅ Define Testing Function
def test(model, test_input, test_output):
    with torch.no_grad():
        test_input = torch.tensor(test_input, dtype=torch.long)
        test_output = torch.tensor(test_output, dtype=torch.float)

        predictions = torch.sigmoid(model(test_input).squeeze(1))
        accuracy = ((predictions > 0.5) == test_output).float().mean().item()
        return accuracy * 100

# ✅ Initialize Model & Train
model = SpamClassifier(vocab_size=len(vocab), embedding_dim=10)
model.embedding.weight.data[w2i['<unk>']] *= 0  # Set unknown token to zero

for i in range(3):
    model = train(model, train_data, train_target, epochs=1)
    print(f"% Correct for test data: {test(model, test_data, test_target):.2f}%")
