# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:04:07 2017

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
from sklearn.metrics import accuracy_score, adjusted_rand_score, calinski_harabaz_score
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

#1.3.1
clf = KMeans(n_clusters=2)
clf.fit(train_data_x)
label_cluster = clf.labels_

clf1 = PCA(n_components=2)
sample_new = clf1.fit_transform(train_data_x)
plt.figure(1)
plt.scatter(sample_new[label_cluster == 1, 0], sample_new[label_cluster == 1, 1], color = 'navy', label = 'class 1')
plt.scatter(sample_new[label_cluster != 1, 0], sample_new[label_cluster != 1, 1], color = 'turquoise', label = 'class 2')
plt.title("PCA transformation on training sample with different colors for K-Means")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()

#1.3.2
clf = AgglomerativeClustering(n_clusters=2)
clf.fit(train_data_x)
label_cluster_ag = clf.labels_

clf2 = PCA(n_components=2)
sample_new = clf2.fit_transform(train_data_x)
plt.figure(2)
plt.scatter(sample_new[label_cluster_ag == 1, 0], sample_new[label_cluster_ag == 1, 1], color = 'navy', label = 'class 1')
plt.scatter(sample_new[label_cluster_ag != 1, 0], sample_new[label_cluster_ag != 1, 1], color = 'turquoise', label = 'class 2')
plt.title("PCA transformation on training sample with different colors(Agglomerative)")
plt.xlabel("feature 1")
plt.ylabel("feature 2")
plt.legend()

#1.3.3
label_cluster[label_cluster == 0] = -1
label_cluster_ag[label_cluster_ag == 0] = -1

print ("Adjusted Rand Index:")
print ("Performance of K-Means:", adjusted_rand_score(train_data_y, label_cluster))
print ("Performance of Agglomerative Clustering:", adjusted_rand_score(train_data_y, label_cluster_ag))

print ("Calinski-Harabaz Index:")
print ("Performance of K-Means:", calinski_harabaz_score(train_data_x, label_cluster))
print ("Performance of Agglomerative Clustering:", calinski_harabaz_score(train_data_x, label_cluster_ag))