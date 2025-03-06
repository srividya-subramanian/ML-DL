# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:20:07 2025

@author: srivi
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import os


path ='C:/Users/srivi/Documents/ML_data/Sherlock_Holmes.txt'
text = open(path, 'r', encoding = "utf8").read().lower()
print('length of the corpus is: :', len(text))


lines = []
for i in text:
    lines.append(i)

data = ""
for i in lines:
    data = ''.join(lines)

data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')
data = data.split()
data = ' '.join(data)