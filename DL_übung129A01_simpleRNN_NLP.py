# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 05:36:34 2025

@author: srivi
"""

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# as in the book :
# a) Generate random word embeddings for a given vocabulary
#def generate_random_word_embeddings(vocabulary, embedding_size=50):
#    return {word: np.random.rand(embedding_size) for word in vocabulary}

def load_glove_embeddings(glove_path, embedding_size=50):
    """Loads GloVe embeddings from a file into a dictionary."""
    word_embeddings = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            word_embeddings[word] = vector
    return word_embeddings

def get_word_embeddings(vocabulary, glove_embeddings, embedding_size=50):
    """Fetches embeddings for words in the vocabulary; uses random vectors for missing words."""
    word_vectors = {}
    for word in vocabulary:
        if word in glove_embeddings:
            word_vectors[word] = glove_embeddings[word]
        else:
            word_vectors[word] = np.random.rand(embedding_size)  # Random vector for unknown words
    return word_vectors

# b) Compute the average of word vectors for a sentence
def average_word_vectors(sentence, word_embeddings, embedding_size=50):
    words = sentence.lower().split()
    word_vectors = [word_embeddings[word] for word in words if word in word_embeddings]

    if len(word_vectors) == 0:  # Handle unseen words
        return np.zeros(embedding_size)
    
    return np.mean(word_vectors, axis=0)

# c) Multiply averaged word vector by an identity matrix
def apply_identity_matrix(word_vector):
    return np.dot(np.identity(len(word_vector)), word_vector)

# d) Create a simple RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, 50)))  # 1 time step, 50 features (word vector size)
model.add(Dense(1, activation='sigmoid'))  # Binary classification (0 = negative, 1 = positive)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# e) Train the RNN model on sentences with sentiment labels


# Load pre-trained GloVe embeddings (Make sure to download a GloVe file, e.g., glove.6B.50d.txt)
glove_path = "C:/Users/srivi/Documents/ML_data/glove.6B.50d.txt"  # Update with the correct path to your GloVe file
glove_embeddings = load_glove_embeddings(glove_path)

# Define vocabulary
vocabulary = ["amazing", "boring", "exciting", "awful", "great", "worst", "best", "enjoyable", "decent"]

# Get word embeddings
word_embeddings = get_word_embeddings(vocabulary, glove_embeddings)

# Print an example
print("Embedding for 'amazing':", word_embeddings["amazing"])


sentences = [
    "Amazing movie, I loved it!",
    "Terrible film, waste of time.",
    "It was okay, not the best but enjoyable.",
    "Great performances and story!",
    "Worst movie ever, do not watch.",
    "Decent film with good acting.",
    "Loved the cinematography and music!",
    "Awful, boring and slow.",
]

# Convert ratings into binary labels (0 = Negative, 1 = Positive)
ratings = [1, 0, 1, 1, 0, 1, 1, 0]  

# Convert sentences into input vectors
training_data = np.array([apply_identity_matrix(average_word_vectors(sentence, word_embeddings)) for sentence in sentences])
training_data = training_data.reshape(len(training_data), 1, 50)  # Reshape to (samples, time_steps, features)
training_labels = np.array(ratings)

# Train the model
model.fit(training_data, training_labels, epochs=10, batch_size=2)

# f) Test the model with new sentences
test_sentences = ["The film was great", "It was a terrible movie"]
test_data = np.array([apply_identity_matrix(average_word_vectors(sentence, word_embeddings)) for sentence in test_sentences])
test_data = test_data.reshape(len(test_data), 1, 50)  # Reshape for RNN input

# Make predictions
predictions = model.predict(test_data)
print("Predicted :", predictions)
