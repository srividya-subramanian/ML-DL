# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 10:32:31 2025

@author: srivi
"""

Generate a random letter using a string and a random module


# Import string and random module
import string
import random

# Randomly choose a letter from all the ascii_letters
randomLetter = random.choice(string.ascii_letters)
print(randomLetter)


Methods of Sentence Embedding
Several methods are employed to generate sentence embeddings:

Averaging Word Embeddings: 
This approach takes the average word embeddings 
within a sentence. While simple, it may not capture complex contextual nuances.

Pre-trained Models like BERT: Models like BERT (Bidirectional Encoder 
Representations from Transformers) have revolutionized sentence embeddings. 
BERT-based models consider each word’s context in a sentence, resulting in rich 
and contextually aware embeddings.

Neural Network-Based Approaches: Skip-Thought vectors and InferSent are examples 
of neural network-based sentence embedding models. They are trained to predict 
the surrounding sentences, encouraging them to understand sentence semantics.

Sentence Embedding Libraries
Like word embedding, sentence embedding is a popular research area. Its interesting techniques break the barrier and help machines understand our language.

Doc2Vec
SentenceBERT
InferSent
Universal Sentence Encoder


