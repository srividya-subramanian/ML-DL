# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:05:56 2025

@author: srivi
"""

'''Explain how the introduction of conditional correlation and the use of the 
Rectified Linear Unit (ReLU) function improves the ability of a deep neural network 
to recognize complex patterns. '''
#    Correlation: normally hidden middle layer is correlated with the 
#    input layer through weights - for example: x% to one input, y% to another and 
#    z% to the other.
#   Conditional correlation: makes the middle nodes be x% correlated to one input
#   when it wants to be and otherwise remain uncorrelated - for example: swithching
#   off nodes when it would be negative. By tuning the middle node like this is 
#   not possible in 2-layer neural network (NN). This logic makes the NN non-linear,
#   otherwise it would be linear. This one is the simplest form of introducing 
#   non-linearity and its called Relu (Rectified Linear Unit). The nonlinearity 
#   of activation functions is crucial for the success of predictive models and
#   enhance accuracy by enabling the system to learn parameters efficiently. The 
#   system learns to making conditional decisions.  

'''Use an example in which a neural network is trained 
to distinguish between images of cats and dogs. Include the following points: '''
    
import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds  # collection of large datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/"
test_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/test_set/"
train_cat_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/cats/"
test_dog_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/test_set/dogs/"

import glob
from pathlib import Path
train_cat_files = Path(train_dataset_path).glob('cats/*.jpg')
train_dog_files = Path(train_dataset_path).glob('dogs/*.jpg')
test_cat_files = Path(test_dataset_path).glob('cats/*.jpg')
test_dog_files = Path(test_dataset_path).glob('dogs/*.jpg')

import matplotlib.pyplot as plt
tf_keras = tf.keras

train_cat_data = [tf_keras.preprocessing.image.load_img(img) for img in train_cat_files]
test_cat_data =  [tf_keras.preprocessing.image.load_img(img) for img in test_cat_files]
train_dog_data = [tf_keras.preprocessing.image.load_img(img) for img in train_dog_files]
test_dog_data =  [tf_keras.preprocessing.image.load_img(img) for img in test_dog_files]
train_cat_data = train_cat_data[0:4000]
test_cat_data = test_cat_data[0:1010]
train_dog_data = train_dog_data[0:4000]
test_dog_data = test_dog_data[0:1010]

print("train size: {} cats and {} dogs".format(len(train_cat_data), len(train_dog_data)))
print("test size :  {} cats and  {} dogs".format(len(test_cat_data), len(test_dog_data)))
    
datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
WIDTH = 128
HEIGHT = 128
IMG_SIZE = (WIDTH , HEIGHT)
BATCH = 32
train_generator = datagen.flow_from_directory(train_dataset_path, target_size = IMG_SIZE,
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    save_format="jpg",
                                                    seed = 1
                                                    )
test_generator = datagen.flow_from_directory(test_dataset_path, target_size = IMG_SIZE,
                                                    class_mode='binary',
                                                    batch_size=BATCH,
                                                    save_format="jpg",
                                                    seed = 1
                                                    )

print(train_generator.class_indices)
print(train_generator.num_classes)
print(train_generator.samples)

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D,Flatten,Dense
model = Sequential()
model.add(Conv2D(16, 3,activation='ReLU', data_format="channels_last", padding='same'))#, input_shape=(48,48,1)))
model.add(Conv2D(32, 3, padding='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(2))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(train_generator, epochs=10,steps_per_epoch = 400)

'''(a) Describe how a neural network without nonlinearities such as the ReLU function 
could only recognize linear patterns and why this is not sufficient for distinguishing 
between cats and dogs. 

b) Explain how the ReLU function is implemented in the network and how it helps 
    to introduce nonlinearities. 

c) Explain how the conditional correlation is enabled by the ReLU function and why 
    this helps the network to learn more complex patterns such as the distinction
    between cats and dogs. 

d) Create a simple Python script that shows the initialization of a small neural 
    network with a hidden layer, applies the ReLU function and illustrates the 
    conditional correlation.  

'''
