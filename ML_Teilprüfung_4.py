# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 13:52:46 2025

@author: srivi
"""

'''NumPy array from the given data set'''
import numpy as np
X = [1.5, 3.5, 5.1, 6.2, 7.9, 2.3, 4.6, 6.3, 7.5, 3.4]
y = [2.4, 4.2, 5.8, 7.4, 8.2, 3.5, 4.9, 6.1, 7.9, 4.0]

data = np.array([list(x) for x in zip(X, y)])  


''' K-Means algorithm from `sklearn.cluster` to divide the data into 3 clusters'''
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init="k-means++", n_init="auto", max_iter=300, random_state=42)
y_pred = km.fit_predict(data)


'''Draw the resulting clusters and their centers'''
colors = ['lightgreen', 'lightpink', 'lightblue']
import matplotlib.pyplot as plt
for i in range(3):
    plt.scatter(data[y_pred == i,0], data[y_pred == i,1], s=100, c=colors[i],edgecolors='black')
    cc= data[y_pred==i].mean(axis=0)
    plt.scatter(cc[0],cc[1],s=80, marker='*', c='black')
    
plt.show()
    
