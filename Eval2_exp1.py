# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 18:42:24 2017

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
alpha_list = [1, 0.5, 0.1, 0.05, 0.01, 0.001]

count = 0

for alpha_val in alpha_list:
    lasso = Lasso(alpha = alpha_val)
    lasso.fit(train_data_x, train_data_y)
    lasso_pred = lasso.predict(test_data_x)
    threshold = 0
    lasso_pred = (lasso_pred > threshold).astype(int)
    lasso_pred[lasso_pred == 0] = -1
    accuracy = accuracy_score(test_data_y, lasso_pred) * 100
    acc_test.append(accuracy)
#    print ("Accuracy of lasso", accuracy)
    if(count%2 == 0):
        coefs = np.r_[lasso.intercept_,lasso.coef_]  
        plt.bar(range(len(coefs)),coefs) 
        plt.title('Testing Accuracy = %2.2f' % acc_test[count]) 
        plt.xlabel('Attribute Index')
        plt.ylabel('Regression Coefficients')
        plt.show()
    count += 1
    
plt.plot(acc_test,'ro-') 
plt.xticks(range(len(alpha_list)),alpha_list)  
plt.xlabel('Regularization Coefficient $\lambda$') 
plt.ylabel('Testing Accuracy') 
plt.show() 
 