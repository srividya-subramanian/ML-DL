# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 20:20:24 2025

@author: srivi
"""

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

# Load data from intents.json
'''Step 2: Import and Load Data File'''
data_file = open('/Users/srivi/Documents/ML_data/intents.json').read()
documents = json.loads(data_file)


'''Step 3: Preprocess Data '''
from nltk.corpus import stopwords
# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

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
# Sample data
data = [
    "The quick brown fox jumps over the lazy dog",
    "A bird in the hand is worth two in the bush",
    "Actions speak louder than words"
]

# Tokenize, lemmatize, and remove stopwords
tokenized_data = []
for sentence in data:
    tokens = nltk.word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatize tokens
    filtered_tokens = [token for token in lemmatized_tokens if token not in stop_words]  # Remove stop words
    tokenized_data.append(filtered_tokens)

# Remove duplicate words
for i in range(len(tokenized_data)):
    tokenized_data[i] = list(set(tokenized_data[i]))

print(tokenized_data)

'''Step 4: Create Training and Testing Data'''
# Create training data

training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in pattern_words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert to numpy array
random.shuffle(training)
training = np.array(training)

# Separate features and labels
train_x = list(training[:,0])
train_y = list(training[:,1])


'''Step 5: Build the Model'''
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

'''Step 6: Predict the Response'''

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


'''Step 7: Run the Chatbot'''

# GUI with Tkinter

import tkinter
from tkinter import *

# Function to send message and get response
def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
# GUI setup

base = Tk()
base.title("Chatbot")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5, bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff', command= send )
EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)
base.mainloop()


'''Write a python program to implement Simple Chatbot?'''

greetings = ["hi", "hello", "hey"]
questions = ["how are you", "what's up?"]
responses = ["I'm doing well, thanks for asking!", "Just hanging out, waiting to chat!"]
farewells = ["bye", "goodbye", "see you later"]


def respond(message):
    message = message.lower()
    if message in greetings:
        return "Hey there!"
    elif message in questions:
        return responses[0]  # Choose a random response from responses list
    elif message in farewells:
        return "See you later!"
    else:
        return "I don't understand, but it sounds interesting!"

# Main loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    print("Chatbot:", respond(user_input))






