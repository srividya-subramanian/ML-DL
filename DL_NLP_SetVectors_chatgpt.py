# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 05:40:36 2025

@author: srivi


In Natural Language Processing (NLP), "set vectors" typically refer to word 
embeddings or sentence vectors that represent words, phrases, or entire documents
 as numerical vectors in a high-dimensional space. These vectors help capture 
 semantic meaning and relationships between words.

1. Word Embeddings (Word Vectors)
Word embeddings represent words as continuous-valued vectors, allowing NLP models
to understand semantic similarity. Some common techniques include:

✅ One-Hot Encoding

Each word is represented as a binary vector of size equal to the vocabulary.
Example for a vocabulary: ["sky", "blue", "cloud", "rain", "sun"]

"sky"   → [1, 0, 0, 0, 0]  
"blue"  → [0, 1, 0, 0, 0]  
"cloud" → [0, 0, 1, 0, 0]  
✅ Word2Vec (CBOW & Skip-gram)

Generates dense, meaningful word vectors by training on context words.
Words with similar meanings have closer vector representations.

✅ GloVe (Global Vectors for Word Representation)

Uses a matrix factorization approach based on word co-occurrence statistics.

2. Sentence or Document Vectors
Instead of individual word vectors, sentences or documents can be converted into 
fixed-length vectors.

✅ Sentence Embeddings (Sentence2Vec, BERT, USE)

Average word embeddings to get a sentence vector.
Use pretrained models like BERT, Universal Sentence Encoder (USE), or InferSent.
✅ TF-IDF Weighted Word Embeddings

Compute word embeddings and weight them using TF-IDF scores before averaging.
✅ Transformer-based Embeddings

Models like BERT, GPT, and T5 generate contextualized embeddings, meaning the 
same word can have different vectors depending on context.

 Applications of Word Vectors in NLP
✅ Text Classification (Sentiment Analysis, Spam Detection)
✅ Machine Translation (Google Translate, DeepL)
✅ Named Entity Recognition (NER) (Recognizing people, places, organizations)
✅ Question Answering & Chatbots
✅ Semantic Search & Information Retrieval

a. It helps to preserve the dominant meaning of a sentence.

Averaging word embeddings combines individual word vectors into a single 
representation, capturing the overall semantic essence of the sentence. 
This method provides a general sense of the sentence's meaning by blending the 
meanings of its constituent words. 


b. It causes similar sentences to have similar vectors.

By averaging word embeddings, sentences with similar content yield similar vector 
representations. This property is beneficial for tasks like clustering or 
classifying sentences with related meanings. 

c. All sentence vectors are the same size, which makes comparison easier.

Representing sentences as fixed-size vectors simplifies their comparison. #
Fixed-size sentence embeddings facilitate the application of various similarity 
measures, such as cosine similarity, to assess the relationships between sentences. 
This uniformity is particularly advantageous in tasks like semantic similarity 
assessments, information retrieval, and clustering, where consistent vector 
dimensions are essential for effective processing.


Neural networks utilize embeddings to capture and represent complex patterns 
within data, enabling the recognition of correlations with target labels. 
These embeddings are learned representations that map input features—such 
as words in natural language processing—into continuous vector spaces. In these 
spaces, the geometric relationships among vectors reflect semantic or categorical 
similarities, which the network leverages to make predictions.

During training, the network adjusts the embeddings to position semantically or 
categorically similar inputs closer together in the vector space. This spatial 
arrangement allows the network to discern patterns and correlations between inputs 
and their corresponding target labels. For instance, in natural language processing, 
embeddings can capture the contextual relationships between words, aiding in tasks 
like word prediction or sentiment analysis.

It's important to note that while embeddings effectively capture these relationships,
 the specific "curve shapes" of the embeddings are not the primary focus. Instead, 
 it's the relative positions and distances between vectors in the embedding space 
 that encode meaningful information about the data and its correlation with target labels.

In summary, neural networks employ embeddings to map input features into vector 
spaces where the spatial relationships among vectors encapsulate the correlations
 with target labels, facilitating accurate predictions.
 
"""
