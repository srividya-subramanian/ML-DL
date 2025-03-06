# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 22:02:39 2025

@author: srivi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:42:07 2025

@author: srivi
"""

import numpy as np
import re
'''
Create a Python script that initializes the initial word embeddings for a 
vocabulary of five words as null vectors. Use a dictionary with the keys as 
words and the values as 3-dimensional NumPy arrays. 
'''

# Build vocabulary
# vocabulary of five words related to sky, as the test sample is “sky blue clouds”
vocab = ["blue", "cloudy", "sunny", "rainy", "sky"]
vocab_index = {word: i for i, word in enumerate(sorted(vocab))}  # Assign index

# Initialize word embeddings as 3-dimensional null vectors
# dictionary with the keys as words and the values as 3-dimensional NumPy arrays

dim = 3
word_embeddings = {word: np.random.rand(dim) * 0.01 for word in vocab}  # Small random values

'''
Define a transition matrix as a unit matrix and an output matrix (sent2output) 
with random values, which serves as a classification layer for predicting the next word. 
'''
# transition matrix as a unit (identity) matrix
transition_matrix = np.eye(dim)

# output matrix (sent2output) with random values
sent2output = np.random.rand(dim, len(vocab)) 

'''
Implement a forward propagation function that takes a given sentence 
(e.g. “sky blue clouds”) and generates the sentence vectors by summing the word 
embeddings and multiplying by the transition matrix. Use the softmax function 
to predict the probability of the next word. 
'''
# Activation function : softmax 
# Softmax function for normalizing outputs
def softmax_activationfn(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability trick
    return exp_x / np.sum(exp_x)


# Tokenization function
def tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = text.split()  # Split into words
    return tokens

# Implement Forward Propagation
def forward_propagation(x, wi, wo):
    h = np.dot(x, wi)  # Hidden state computation
    y_pred = softmax_activationfn(np.dot(h, wo))  # Compute softmax output
    return h, y_pred

'''
Write a backpropagation function that calculates the error between the prediction 
and the actual next word (e.g. “sun”) and uses the gradients to update the word 
embeddings, the transition matrix and the output matrix. 
'''

# Implement Backpropagation
def back_propagation(h, x, y, y_pred, lr, sent2output, transition_matrix, word_embeddings, words):
    err = y_pred - y  # Compute the error
    grad_o = np.outer(h, err)  # Gradient for output matrix
    grad_h = np.dot(sent2output, err)  # Backpropagate error to hidden layer
    grad_tmat = np.outer(grad_h, x)  # Gradient for transition matrix

    # Update matrices
    sent2output -= lr * grad_o
    transition_matrix -= lr * grad_tmat

    # Update word embeddings
    for word in words:
        word_embeddings[word] -= lr * grad_h

    return sent2output, transition_matrix, word_embeddings


# Train the model with a small dataset
training_data = [
    ("sky blue", "cloudy"),
    ("blue cloudy", "rainy"),
    ("rainy sky", "sunny"),
    ("sunny blue", "sky"),
    ("cloudy rainy", "blue")
]

epochs = 1000
learning_rate = 0.01

for epoch in range(epochs):
    total_loss = 0
    for sentence, next_word in training_data:
        words = tokenize(sentence)
        x = sum(word_embeddings[word] for word in words if word in word_embeddings)
        h, y_pred = forward_propagation(x, transition_matrix, sent2output)

        # One-hot encode target word
        target = np.zeros(len(vocab))
        target[vocab_index[next_word]] = 1

        # Compute loss
        loss = -np.sum(target * np.log(y_pred + 1e-8))  # Cross-entropy loss
        total_loss += loss

        # Perform backpropagation
        sent2output, transition_matrix, word_embeddings = back_propagation(
            h, x, target, y_pred, learning_rate, sent2output, transition_matrix, word_embeddings, words
        )

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# Step 6: Test the trained model
test_sentence = "sky blue"
test_words = tokenize(test_sentence)
x_test = sum(word_embeddings[word] for word in test_words if word in word_embeddings)

_, predicted_probs = forward_propagation(x_test, transition_matrix, sent2output)
predicted_index = np.argmax(predicted_probs)
predicted_word = vocab[predicted_index]

print("\nTest Sentence:", test_sentence)
print("Predicted Next Word:", predicted_word)
