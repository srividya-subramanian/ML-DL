# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:11:17 2025

@author: srivi
"""
'''
Why Use Averaged Word Vectors?
✅ Converts variable-length text into fixed-length vectors
✅ Captures semantic meaning of words in a document
✅ Works well for text classification, clustering, and similarity tasks
✅ Simpler than more advanced models like LSTMs or Transformers


🔹 Steps to Compute Averaged Word Vectors
Convert text into word tokens (Tokenization)
Get word embeddings for each token (using Word2Vec, GloVe, FastText, etc.)
Compute the mean (average) of all word vectors


🔹 Applications of Averaged Word Vectors
✅ Text Classification – Convert reviews/emails/documents into feature vectors 
for machine learning
✅ Document Similarity – Compare similarity between two pieces of text
✅ Clustering – Group similar texts (e.g., topic modeling)
✅ Semantic Search – Improve keyword-based search by considering word meanings


🔹 Limitations
⚠ Ignores word order – Simple averaging does not capture syntax or context
⚠ Out-of-Vocabulary (OOV) Words – If a word is not in the embedding model, 
it gets ignored
⚠ Not as powerful as Transformers – More advanced methods like BERT capture 
context-dependent meanings


🔹 Alternative: TF-IDF Weighted Word Vectors
Instead of simple averaging, weigh important words higher using TF-IDF:


🔹 Summary
Method	                Pros	                    Cons
Averaged Word Vectors	Simple, fast, effective	    Ignores word order, context
TF-IDF Weighted Vectors	Considers importance of words	Requires additional computation
Advanced Models (BERT, LSTM)	Captures syntax & context	Computationally expensive

'''





import numpy as np
import gensim.downloader as api
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load Pre-trained Word2Vec Model (Google's Word2Vec)
word2vec = api.load("word2vec-google-news-300")  # 300-dimensional embeddings

# Function to get averaged word vectors
def get_avg_word_vector(text, model):
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    vectors = [model[word] for word in tokens if word in model]  # Get word embeddings
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)  # Return zero vector if no words found
    return np.mean(vectors, axis=0)  # Compute mean vector

# Example sentence
text = "This movie was absolutely fantastic!"
vector = get_avg_word_vector(text, word2vec)

print("Averaged Word Vector Shape:", vector.shape)  # Output: (300,)
print("Averaged Word Vector (first 10 values):", vector[:10])  # Preview first 10 values


from sklearn.feature_extraction.text import TfidfVectorizer

# Compute TF-IDF weights
tfidf = TfidfVectorizer()
tfidf.fit(["This is a sample document", "Another example document"])

# Function to compute TF-IDF weighted word vectors
def get_tfidf_weighted_vector(text, model, tfidf):
    tokens = word_tokenize(text.lower())
    vectors = []
    weights = []
    
    for word in tokens:
        if word in model and word in tfidf.vocabulary_:
            vectors.append(model[word] * tfidf.idf_[tfidf.vocabulary_[word]])
            weights.append(tfidf.idf_[tfidf.vocabulary_[word]])
    
    if len(vectors) == 0:
        return np.zeros(model.vector_size)
    return np.average(vectors, axis=0, weights=weights)  # Weighted average

# Example
vector = get_tfidf_weighted_vector("This movie was fantastic!", word2vec, tfidf)
print("TF-IDF Weighted Word Vector Shape:", vector.shape)








