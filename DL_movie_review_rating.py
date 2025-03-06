# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:48:03 2025

@author: srivi
"""
# Simple Neural Network for Movie Review Rating Prediction

# Tokenization is the process of breaking down a text into individual words or 
# tokens. This is an essential first step before applying Bag of Words (BoW)

# Bag of Words (BoW): Converting Text to Numerical Features
# - converts tokenized text into a fixed-length vector based on word frequency.
# Each unique word in the dataset becomes a feature in a vector, and the value 
# represents the word count in a document.

'''
🔹 Advantages and Limitations of BoW
✅ Advantages

Simple and effective for text classification tasks.
Works well when word frequency matters.
⚠ Limitations

Ignores word order (e.g., “I love this movie” vs. “This movie I love” are 
                    treated the same).
Sparse representation (large vocabularies lead to large vectors with many zeros).
Does not capture meaning or context (e.g., synonyms like “great” and “awesome” 
                                     are treated separately).
🔹 Alternative to BoW: TF-IDF
To improve BoW, we can use TF-IDF (Term Frequency-Inverse Document Frequency), 
which weighs words by importance.

'''
import numpy as np
import re
from collections import Counter
from sklearn.model_selection import train_test_split


# Sample dataset
documents = [
    "Amazing movie, I loved it!",
    "Terrible film, waste of time.",
    "It was okay, not the best but enjoyable.",
    "Great performances and story!",
    "Worst movie ever, do not watch.",
    "Decent film with good acting.",
    "Loved the cinematography and music!",
    "Awful, boring and slow.",
]
ratings = [5, 1, 3, 5, 1, 3, 5, 1]  # 1 to 5 star ratings

def tokenize(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = text.split()  # Split into words
    return tokens

# Tokenize each document
tokenized_docs = [tokenize(doc) for doc in documents]

# Build vocabulary (unique words)
all_words = {word for doc in tokenized_docs for word in doc}
vocab = {word: i for i, word in enumerate(sorted(all_words))}  # Assign index

print("Vocabulary:", vocab)

# Convert each document into a BoW vector
def bow_vector(tokens, vocab):
    vector = [0] * len(vocab)
    word_counts = Counter(tokens)
    for word, count in word_counts.items():
        if word in vocab:
            vector[vocab[word]] = count
    return vector

bow_vectors = [bow_vector(doc, vocab) for doc in tokenized_docs]

# Display BoW vectors
#or i, vec in enumerate(bow_vectors):
#   print(f"Document {i+1}: {vec}")

X = np.array(bow_vectors)
num_classes = 5  # Ratings from 1 to 5
num_vocab = len(vocab)
y_onehot = np.eye(num_classes)[np.array(ratings) - 1]

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

learning_rate = 0.01
epochs = 1000


def initialise(num_vocab, num_classes):
    ipLayerSize, hdLayerSize, opLayerSize = num_vocab,10,num_classes
    wi = np.random.rand(ipLayerSize, hdLayerSize) 
    bi = np.zeros((1, hdLayerSize))
    wo = np.random.rand(hdLayerSize, opLayerSize) 
    bo = np.zeros((1, opLayerSize))
    print(wi.shape, wo.shape, bi.shape, bo.shape)
    return wi, wo, bi, bo


def perceptron_npdot(x, w): 
    y = np.dot(x, w)
    return y

def relu_activationfn(x):
    return np.maximum(x, 0)

def derivative_relu(x): 
    return x > 0

from math import e
def softmax_activationfn(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
    y = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Normalize correctly
    return y

def error(y, y_pred):
    err = y - y_pred
    return err

def cross_entropy_loss(y_pred, y, wi, wo):
    l2_lambda = 0.001
    loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8), axis=1))
    l2_penalty = (l2_lambda / 2) * (np.sum(wi**2) + np.sum(wo**2))  # L2 Regularization term
    return loss

def forward_propagation(x,wi,wo,bi,bo):
    #print(x.shape,wi.shape)
    #print(((perceptron_npdot(x,wi))+bi).shape)
    hx = perceptron_npdot(x,wi)+bi
    ho  = relu_activationfn(hx)
    yh = perceptron_npdot(ho,wo)+bo
    y_pred = softmax_activationfn(yh)
    return y_pred, ho

def back_propagation(h, wi, wo, bi, bo, x, y, y_pred, lr):
    err = y_pred - y  # Correct gradient for softmax
    deltao = err  # No need for ReLU derivative at output layer

    # Hidden layer gradient
    deltah = np.dot(deltao, wo.T) * derivative_relu(h)

    # Weight updates
    wo -= np.dot(h.T, deltao) * lr
    wi -= np.dot(x.T, deltah) * lr
    
    # Bias updates (corrected)
    bo -= np.sum(deltao, axis=0, keepdims=True) * lr
    bi -= np.sum(deltah, axis=0, keepdims=True) * lr

    return wi, wo, bi, bo  # Return updated weights and biases



wi, wo, bi, bo = initialise(num_vocab, num_classes)
lr = 0.3 
epochs = 1000

for epoch in range(epochs): 
    #print(X.shape)
    y_pred, h = forward_propagation(X_train,wi,wo,bi,bo)
    wi, wo, bi, bo = back_propagation(h, wi, wo, bi, bo, X_train, y_train, y_pred, lr)

y_pred, h = forward_propagation(X_test,wi,wo,bi,bo)
y_pred=y_pred#.round(1)#.astype(int)
print(y_test.round(1))
print(y_pred.round(1))













