# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 11:06:36 2025

@author: srivi
"""

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
#import tflearn
import tensorflow as tf
import random


import json
data_file = open('/Users/srivi/Documents/ML_data/intents1.json').read()
intents = json.loads(data_file)
    
words = []
classes = []
documents = []
ignore_words = ['?']
# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)  
    

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)
tx=[]
ty=[]
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    tx.append([bag])
    ty.append([output_row])
    training.append([bag, output_row])
    

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype="object")
tx = np.array(tx)
ty = np.array(ty)
# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini',random_state= 0)
rf.fit(train_x,train_y)
#rf.fit(tx,ty)
#y_pred_rf = rf.predict(X_test)


def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    test=[]
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    print(sentence_words)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    test.append([bag])                
#    return(np.array(bag))                	
    return test

sentence = "can I use mastercard?"
p = bow(sentence, words)
test = np.array(p, dtype="object").reshape(1, -1)
test = list(test)
#p = p.reshape(-1, 1))
print (p)
#print (classes)

print(rf.predict(test))



