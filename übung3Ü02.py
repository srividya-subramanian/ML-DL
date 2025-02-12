# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 12:58:48 2025

@author: srivi
"""

import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
plt.scatter(X_xor[y_xor == 1, 0],X_xor[y_xor == 1, 1],c='b', marker='x',label='1')
plt.show()


from sklearn.pipeline import Pipeline  
from sklearn.svm import SVC
svm= SVC(kernel='rbf', random_state=1,gamma=0.1, C=500)

from sklearn import model_selection as model
X_train, X_test, Y_train,Y_test= model.train_test_split(X_xor,y_xor, test_size=0.2, train_size=0.8)

Pipeline=Pipeline([
   ('classifier',svm)  
    ])

Pipeline.fit(X_train,Y_train)
y_pred = Pipeline.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred))

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_xor, y_xor, clf=svm)
plt.xlabel('Länge des Blütenblatts [cm]')
plt.ylabel('Breite des Blütenblatts [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
