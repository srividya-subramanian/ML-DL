# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:20:59 2025

@author: srivi


a) Die Tabelle könnte folgendermaßen aussehen: 

A screenshot of a cell phone

Description automatically generated 

b) Die One-hot-Codierung wird angewendet, indem für jedes Wort im Vokabular ein eindeutiger Vektor erstellt wird, bei dem nur an der dem Wort zugeordneten Stelle eine '1' steht, ansonsten '0'. Wenn ein Wort mehrmals in einer Bewertung vorkommt, könnte man entweder die Vektoren addieren, was zu einer Häufigkeitszählung führt, oder man verwendet eine binäre Darstellung, bei der das Wort im Vektor nur einmal markiert wird, unabhängig davon, wie oft es vorkommt. 

c) Python-Code für das neuronale Netz könnte so aussehen: 

 
import numpy as np 
from keras.models import Sequential 
from keras.layers import Dense 
 


# Angenommen, `data` ist ein Array von One-hot-codierten Vektoren 

# und `labels` ist ein Array von 0 und 1, die die Stimmung repräsentieren. 

 
model = Sequential() 
model.add(Dense(10, input_dim=len(vocabulary), activation='relu')) 
model.add(Dense(1, activation='sigmoid')) 
 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 
 
model.fit(data, labels, epochs=10, batch_size=1) 
 




# Bewertung des Modells 

loss, accuracy = model.evaluate(data, labels) 
print(f'Accuracy: {accuracy}') 

d) Als Metriken zur Bewertung der Modellleistung würden die Genauigkeit (Accuracy), die F1-Score, Precision und Recall verwendet werden. 

e) Für die Fehleranalyse würde ich die falsch positiven und falsch negativen Vorhersagen untersuchen, um Muster zu erkennen. Basierend auf diesen Erkenntnissen könnte das Modell verbessert werden, indem man zum Beispiel das Vokabular erweitert, die Anzahl der Neuronen in den Schichten anpasst oder eine andere Art von Embedding, wie Word2Vec oder GloVe, verwendet. 

"""