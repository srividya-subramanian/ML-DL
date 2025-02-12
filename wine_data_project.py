# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 14:36:05 2025

@author: srivi
"""

import numpy as np
import pandas as pd
pd.set_option('display.max_columns',None)


'''Read data'''
df = pd.read_csv("/Users/srivi/Downloads/wine+quality/winequality-red.csv", delimiter=';')

Y=df['quality']
X=df.drop(['quality'],axis=1)


'''Split the data set into 70% training data and 30% test data'''
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7,stratify=Y)


'''Find important feature'''
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, criterion='gini',random_state= 0)
rf.fit(X_train,Y_train)
y_pred_rf = rf.predict(X_test)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = df.columns[0:]
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Bedeutung der Merkmale')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()


'''confusion matrix'''
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred_rf)
print(cm)


from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


models = []
#models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=6)))
models.append(('DT', DecisionTreeClassifier(criterion='gini', max_depth=5,random_state= 0)))
models.append(('RF', RandomForestClassifier(n_estimators=1000, criterion='gini',random_state= 0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto',kernel='rbf', random_state=1, C=500)))


# evaluate each model in turn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=7, random_state=0, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()


