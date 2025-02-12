# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 09:22:47 2025

@author: srividya
@Workshop guide : Cornelius Schätz
                  https://github.com/cornelius31415/ARTIFICIAL-INTELLIGENCE/tree/main/CHATBOT

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''IMPORT DATA'''
df = pd.read_csv("/Users/srivi/Downloads/mushroom.csv")#, delimiter=';')
pd.set_option('display.max_columns', None)

'''DATA DESCRIPTION'''
print(df.head(5))
print(df.describe())

'''SPLIT THE DATA'''
y = df['class']
X = df.drop('class', axis =1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, train_size=0.7,stratify=y)

'''RANDOMFOREST'''
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, criterion='gini',random_state= 0)
rf.fit(X_train,Y_train)
y_pred_rf = rf.predict(X_test)


'''Heatmap of Feature Correlation'''
from mlxtend.plotting import heatmap
cols = X.columns[0:]
cm = np.corrcoef(df[cols].values.T)
heatmap(cm, column_names=cols, row_names=cols, cell_font_size = 2)
plt.title('Heatmap of Feature Correlation')
plt.show()

'''RF Feature Importance'''
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = cols
from sklearn.metrics import accuracy_score
acc = np.round(accuracy_score(Y_test, y_pred_rf),3)
print('------------------------------------')
print('Accuracy of RF: ', acc)
print('------------------------------------')

for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
plt.title('Feature Importance')
plt.bar(range(X_train.shape[1]), importances[indices], align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=45)
plt.xlim([-1, X_train.shape[1]])
plt.show()

'''Various models selected for training and predition '''
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

'''CREATE A COMBINED MODEL'''
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
models.append(('DT', DecisionTreeClassifier(criterion='gini', max_depth=5,random_state= 0)))
models.append(('RF', RandomForestClassifier(n_estimators=100, criterion='gini',random_state= 0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto',kernel='rbf', random_state=1, C=100)))

'''EVALUATE EACH MODEL'''
# evaluate each model in turn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))
# Compare Algorithms
plt.boxplot(results, tick_labels=names)
plt.title('Algorithm Comparison')
plt.show()

'''PRINT CLASSIFICATION REPORT OF THE BEST CLASSIFIER: RF'''
from sklearn.metrics import classification_report
print(classification_report(Y_test, y_pred_rf))


'''Confusion matrix of RF classifier'''
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
cm = confusion_matrix(Y_test, y_pred_rf)
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
cm_display.plot()
plt.title('Confusion matrix of RF classifier')
plt.show()
