# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 20:04:52 2017

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
from sklearn.metrics import accuracy_score, adjusted_rand_score
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
acc_test = []
acc_train = []
gamma_list = [1e-5,1e-3,1e-2,1e-1,1,1e1,1e2] 

for gamma_val in gamma_list:
    svm1 = svm.SVC(C = 1, kernel = 'rbf', gamma = gamma_val)
    svm1.fit(train_data_x, train_data_y)
    svm_pred1 = svm1.predict(train_data_x)
    svm_pred2 = svm1.predict(test_data_x)
    svm_pred1[svm_pred1 == 0] = -1
    accuracy = accuracy_score(train_data_y, svm_pred1) * 100
    acc_train.append(accuracy)
    accuracy = accuracy_score(test_data_y, svm_pred2) * 100
    acc_test.append(accuracy)
#    print ("accuracy of svm", accuracy)
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(acc_train,'bo-',label='Training Accuracy') 
ax.plot(acc_test,'ro-',label='Testing Accuracy') 
plt.xticks(range(len(gamma_list)),gamma_list) 
plt.xlabel('SVM RBF Kernel Width $\gamma$') 
plt.ylabel('Prediction Accuracy')
ax.legend() 
plt.show()    