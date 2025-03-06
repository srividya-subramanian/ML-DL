# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:50:45 2025

@author: srivi
"""

'''
TF-IDF (Term Frequency-Inverse Document Frequency)

TF-IDF is a numerical statistic that measures how important a word is in a 
document relative to a collection of documents (corpus). It is commonly used 
in text mining, search engines, and NLP tasks such as document classification 
and keyword extraction.

🔹 TF-IDF Formula
TF-IDF consists of two main components:

1️⃣ Term Frequency (TF)
Measures how often a word appears in a document.

𝑇𝐹(w)=Number of times word appears in the document / Total number of words 
                                                        in the document
 
Common words (e.g., "the", "is") will have high TF but may not be meaningful.

2️⃣ Inverse Document Frequency (IDF)
Measures how rare a word is across all documents.

IDF(w)=log( (Total number of documents / Number of documents )+1)
                                       containing the word w

Rare words (e.g., "quantum", "neural") will have high IDF, making them more important.


3️⃣ TF-IDF Score
The final TF-IDF score is calculated as:

TF−IDF(w)=TF(w)×IDF(w)
Higher TF-IDF → The word is important in a specific document.
Lower TF-IDF → The word is common and less useful for distinguishing documents.

🔹 When to Use TF-IDF?
✅ Text Classification – Convert text into numerical features for ML models
✅ Search Engines – Rank documents based on keyword relevance
✅ Keyword Extraction – Identify important words in a document
✅ Spam Filtering – Identify unusual words in emails

🔹 Limitations of TF-IDF
⚠ Ignores Word Order – Cannot capture phrases like "New York" vs. "York New"
⚠ Does Not Handle Synonyms – Treats "car" and "automobile" as different words
⚠ Sensitive to Long Documents – Long documents may have lower TF-IDF scores

✅ Solution → Use Word Embeddings (Word2Vec, GloVe, BERT) for context-aware 
representations.

Would you like an example of TF-IDF with cosine similarity for document comparison? 😊🚀


'''


from sklearn.feature_extraction.text import TfidfVectorizer

# Sample corpus
documents = [
    "The movie was great and full of suspense",
    "The movie was terrible and boring",
    "I loved the movie, it was amazing!"
]

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to readable format
feature_names = vectorizer.get_feature_names_out()
tfidf_array = tfidf_matrix.toarray()

# Print TF-IDF scores
import pandas as pd
df = pd.DataFrame(tfidf_array, columns=feature_names)
print(df)


















