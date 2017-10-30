# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 20:19:21 2017

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
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.cluster import KMeans, AgglomerativeClustering

dataset = np.genfromtxt('C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/phishing_large.csv' , delimiter = ',')
dataset = dataset[0:, :]


X = dataset[:,0:-1]
Y = dataset[:,-1]

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(X,Y,test_size = .25, random_state = 0)

lambda_list = [1, 0.1, 0.01]

for lambda_val in lambda_list:
    idxfold = 0
    kf = KFold(n_splits=4)
    acc_test = np.zeros(4) 
    
    for idxtrain, idxtest in kf.split(dataset):
        sample_train = dataset[idxtrain,0:-1]  
        label_train = dataset[idxtrain,-1]  
        sample_test = dataset[idxtest,0:-1]  
        label_test = dataset[idxtest,-1]  
        clf = Lasso(alpha=lambda_val) 
        clf.fit(sample_train, label_train) 
        label_pred = clf.predict(sample_test) 
        thres = 0
        label_pred = (label_pred>thres).astype(int)
        label_pred[label_pred==0] = -1
        acc_test[idxfold] = accuracy_score(label_test, label_pred) 
        idxfold = idxfold + 1  
    acc_test_mean = np.mean(acc_test) 
    acc_test_var = np.var(acc_test) 
    print("For lambda value ", lambda_val,", the accuracy score is ", acc_test,", acc_test_mean is ", acc_test_mean," and acc_test_var is ",acc_test_var)