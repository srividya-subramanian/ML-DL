# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:27:31 2025

@author: srivi
"""

from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200,noise=0.05,random_state=0)


import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1])
plt.tight_layout()
plt.show()



from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,random_state=0)
y_km = km.fit_predict(X)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,3))
ax1.scatter(X[y_km==0,0],X[y_km==0,1],
            c='lightblue',edgecolor='black',marker='o',s=40,label='Cluster 1')
ax1.scatter(X[y_km==1,0],X[y_km==1,1],
            c='red',edgecolor='black',marker='s',s=40,label='Cluster 2')
ax1.set_title('k-Means-Clustering')



from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2, metric='euclidean',linkage='complete')
y_ac = ac.fit_predict(X)

ax2.scatter(X[y_ac==0,0],X[y_ac==0,1],
            marker='o',s=40,c='lightblue',edgecolor='black', label='Cluster 1')
ax2.scatter(X[y_ac==1,0], X[y_ac==1,1],
            marker='s',s=40,c='red',edgecolor='black', label='Cluster 2')
ax2.set_title('Agglomeratives Clustering')
plt.legend()
plt.tight_layout()
plt.show()



from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.2, min_samples=5,metric='euclidean')
y_db = db.fit_predict(X)
plt.scatter(X[y_db==0,0], X[y_db==0,1],
            marker='o',s=40,c='lightblue',edgecolor='black',label='Cluster 1')
plt.scatter(X[y_db==1,0],X[y_db==1,1],
            marker='s',s=40,c='red',edgecolor='black',label='Cluster 2')
plt.legend()
plt.show()
