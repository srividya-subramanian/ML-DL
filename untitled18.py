# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 12:49:57 2025

@author: srivi
"""

from sklearn.datasets import make_blobs
X,y=make_blobs(n_samples=[100,100,100],centers=[[-3,0],[3,3],[6,-3]],n_features=2,
               cluster_std=[0.3, 0.7, 1.2],random_state=42)


colors=["lightgreen","lightblue","lightpink"]
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1],marker='o',s=20,c='green',edgecolors='black')
plt.show()


from sklearn.cluster import DBSCAN

db = DBSCAN(eps=1.2, min_samples=10, metric="euclidean")
y_db = db.fit_predict(X)

plt.scatter(X[y_db==0,0],X[y_db==0,1],marker='o',s=20,c=colors[0],edgecolors='black',label='Cluster 1')
plt.xlim(-5,10)
plt.ylim(-7,6)
plt.scatter(X[y_db==1,0],X[y_db==1,1],marker='o',s=20,c=colors[1],edgecolors='black',label='Cluster 2')
plt.scatter(X[y_db==2,0],X[y_db==2,1],marker='o',s=20,c=colors[2],edgecolors='black',label='Cluster 3')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('DBScan Clustering')
plt.legend()
plt.show()

