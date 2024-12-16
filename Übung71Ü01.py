# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 08:03:15 2024

@author: srivi
"""
from PIL import Image
import numpy as np


PFAD = 'flower_photos/sunflowers/9588522189_db6166f67f.jpg'
image = Image.open(PFAD)
image = image.resize(size=(28, 28))
image.show()

data = image.getdata()
datalist = list(data)
greyscale = [0]*np.size(datalist)
i=0
for pixel in datalist: 
    new = int(255 - sum(pixel)/3)
    greyscale[i]= new
    i=i+1
pixelcount = np.sum(1 for pixel in greyscale if pixel > 200) 
print(pixelcount)

