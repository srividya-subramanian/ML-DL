# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 17:14:24 2025

@author: srivi
"""

import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds  # collection of large datasets
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

train_cat_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/cats/"
train_dog_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/dogs/"
test_cat_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/test_set/cats/"
test_dog_dataset_path = "C:/Users/srivi/Documents/ML_data/cat_dog/test_set/dogs/"

FAST_RUN = False
IMAGE_WIDTH=128
IMAGE_HEIGHT=128
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3

filenames = os.listdir(train_cat_dataset_path)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df1 = pd.DataFrame({
    'filename': filenames,
    'category': categories
})
filenames = os.listdir(train_dog_dataset_path)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df2 = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df_m = pd.concat([df1, df2], ignore_index=True, sort=False).drop([8006]).sample(frac = 1).reset_index()

TRAIN_DIR = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/"
TRAIN_DIR = "C:/Users/srivi/Documents/ML_data/cat_dog/training_set/"
cats = os.listdir(TRAIN_DIR + "/cats")
dogs = os.listdir(TRAIN_DIR + "/dogs")
DATA_DIR = "C:/Users/srivi/Documents/ML_data/cat_dog/"

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

from torchvision import transforms
from torchvision.datasets import ImageFolder
transform = transforms.Compose([transforms.Resize((112,112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean, std)])
dataset = ImageFolder(DATA_DIR+'/training_set', transform=transform)
test_dataset = ImageFolder(DATA_DIR+'/test_set', transform=transform)


import matplotlib.pyplot as plt

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img)
    #plt.imshow(img.permute(1, 2, 0))
    plt.show()
show_example(*dataset[4800])

from torch.utils.data import Dataset, DataLoader
batch_size=64
train_loader = DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size*2, num_workers=4, pin_memory=True)





