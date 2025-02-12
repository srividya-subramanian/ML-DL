# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 18:16:16 2025

@author: srivi
"""


from os import listdir,makedirs
from numpy import asarray
from numpy import save
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

folder = "C:/Users/srivi/Documents/ML_data/dogs-vs-cats/train/"
import skimage

photos, labels = list(), list()
# enumerate files in the directory
for file in listdir(folder):
	# determine class
	output = 0.0
	if file.startswith('dog'):
		output = 1.0
	# load image
	photo = load_img(folder + file, target_size=(56,56))
	greyphoto = skimage.color.rgb2gray(photo)
    # convert to numpy array
	# photo = img_to_array(greyphoto)
	# store
	photos.append(greyphoto)
	labels.append(output)
# convert to a numpy arrays
photos = asarray(photos)
labels = asarray(labels)
print(photos.shape, labels.shape)
# save the reshaped photos
save('dogs_vs_cats_photos2_56x56.npy', photos)
save('dogs_vs_cats_labels2_56x56.npy', labels)


import matplotlib.pyplot as plt
plt.imshow(photo, cmap='Greys') #8
plt.show() #9








