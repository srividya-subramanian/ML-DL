# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 19:48:01 2024

@author: srivi
"""

from PIL import Image
PFAD = 'bilder/4.png'
bild = Image.open(PFAD)
print('Modus:', bild.mode)
print('Größe:', bild.size)


