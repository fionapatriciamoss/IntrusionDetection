# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:23:48 2017

@author: fmoss1
"""

import numpy as np
import pandas as pd
#import sklearn
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.cluster import KMeans, AgglomerativeClustering

dataset = np.genfromtxt('C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/phishing_large.csv' , delimiter = ',')
dataset = dataset[0:, :]


X = dataset[:,0:-1]
Y = dataset[:,-1]

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X,Y,test_size = .25, random_state = 0)


#1.2.1
clf = PCA(n_components=2)
sample_new = clf.fit(train_data_x).transform(train_data_x)
plt.figure(1)
plt.scatter(sample_new[:,0], sample_new[:,1], color = 'navy')
plt.title("PCA transformation on training sample")
plt.xlabel("feature 1")
plt.ylabel("feature 2")

#1.2.2
clf = PCA(n_components=2)
sample_new = clf.fit_transform(train_data_x)
plt.figure(2)
plt.scatter(sample_new[train_data_y == 1, 0], sample_new[train_data_y == 1, 1], color = 'navy', label = 'class 1')
plt.scatter(sample_new[train_data_y == -1, 0], sample_new[train_data_y == -1, 1], color = 'turquoise', label = 'class 2')
plt.title("PCA transformation on training sample with different colors")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()

#1.2.3 use test instead of train
clf = PCA(n_components=2)
sample_new = clf.fit(train_data_x).transform(test_data_x) 
plt.figure(3)
plt.scatter(sample_new[test_data_y == 1, 0], sample_new[test_data_y == 1, 1], color = 'navy', label = 'class 1')
plt.scatter(sample_new[test_data_y == -1, 0], sample_new[test_data_y == -1, 1], color = 'turquoise', label = 'class 2')
plt.title("PCA transformation on testing sample with different colors")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()

#1.2.4 n_components=15, apply SVM, no plotting
clf = PCA(n_components=15)
train_data = clf.fit(train_data_x)
test_data = clf.transform(test_data_x) 

svm1 = svm.SVC()
svm1.fit(train_data_x, train_data_y)
svm_pred = svm1.predict(test_data_x)
svm_pred[svm_pred == 0] = -1
accuracy = accuracy_score(test_data_y, svm_pred) * 100
print ("accuracy of svm", accuracy)

#1.2.5 All supervised methods
clf = PCA(n_components=15)
train_data = clf.fit(train_data_x)
test_data = clf.transform(test_data_x) 

lm = LinearRegression()
lm.fit(train_data_x, train_data_y)
linear_pred = lm.predict(test_data_x)
threshold = 0
linear_pred = (linear_pred > threshold).astype(int)
linear_pred[linear_pred == 0] = -1
accuracy = accuracy_score(test_data_y, linear_pred) * 100
print ("accuracy of Linear Regression:", accuracy)

ridge = Ridge()
ridge.fit(train_data_x, train_data_y)
ridge_pred = ridge.predict(test_data_x)
threshold = 0
ridge_pred = (ridge_pred > threshold).astype(int)
ridge_pred[ridge_pred == 0] = -1
accuracy = accuracy_score(test_data_y, ridge_pred) * 100
print ("accuracy of ridge regression:", accuracy)

lasso = Lasso()
lasso.fit(train_data_x, train_data_y)
lasso_pred = lasso.predict(test_data_x)
threshold = 0
lasso_pred = (lasso_pred > threshold).astype(int)
lasso_pred[lasso_pred == 0] = -1
accuracy = accuracy_score(test_data_y, lasso_pred) * 100
print ("accuracy of lasso:", accuracy)

logistic = LogisticRegression()
logistic.fit(train_data_x, train_data_y)
logistic_pred = logistic.predict(test_data_x)
threshold = 0
logistic_pred = (logistic_pred > threshold).astype(int)
logistic_pred[logistic_pred == 0] = -1
accuracy = accuracy_score(test_data_y, logistic_pred) * 100
print ("accuracy of logistic regression:", accuracy)

nn = MLPClassifier()
nn.fit(train_data_x, train_data_y)
nn_pred = nn.predict(test_data_x)
threshold = 0
nn_pred = (nn_pred > threshold).astype(int)
nn_pred[nn_pred == 0] = -1
accuracy = accuracy_score(test_data_y, nn_pred) * 100
print ("accuracy of neural networks:", accuracy)

dt = tree.DecisionTreeClassifier()
dt.fit(train_data_x, train_data_y)
dt_pred = dt.predict(test_data_x)
threshold = 0
dt_pred = (dt_pred > threshold).astype(int)
dt_pred[dt_pred == 0] = -1
accuracy = accuracy_score(test_data_y, dt_pred) * 100
print ("accuracy of decision trees:", accuracy)

rf = RandomForestClassifier()
rf.fit(train_data_x, train_data_y)
rf_pred = rf.predict(test_data_x)
threshold = 0
rf_pred = (rf_pred > threshold).astype(int)
rf_pred[rf_pred == 0] = -1
accuracy = accuracy_score(test_data_y, rf_pred) * 100
print ("accuracy of random forest:", accuracy)

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(train_data_x, train_data_y)
kn_pred = kn.predict(test_data_x)
threshold = 0
kn_pred = (kn_pred > threshold).astype(int)
kn_pred[kn_pred == 0] = -1
accuracy = accuracy_score(test_data_y, kn_pred) * 100
print ("accuracy of k nearest neighbours:", accuracy)


#1.2.6
model = KernelPCA(n_components = 15, kernel = 'rbf', gamma = .01)
sample_kpca = clf.fit(train_data_x,train_data_y).transform(train_data_x)

svm2 = svm.SVC(kernel='rbf', C=1, gamma = .01)
svm2.fit(train_data_x, train_data_y)
svm_pred = svm2.predict(test_data_x)
svm_pred[svm_pred == 0] = -1
accuracy = accuracy_score(test_data_y, svm_pred) * 100
print ("accuracy of svm using kpca:", accuracy)

model = LinearDiscriminantAnalysis(n_components = 1)
sample_lda = clf.fit(train_data_x,train_data_y).transform(train_data_x)

svm3 = svm.SVC()
svm3.fit(train_data_x, train_data_y)
svm_pred = svm3.predict(test_data_x)
svm_pred[svm_pred == 0] = -1
accuracy = accuracy_score(test_data_y, svm_pred) * 100
print ("accuracy of svm using lda:", accuracy)














