'''
Created on Sep 30, 2017

@author: fmoss1
'''

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

dataset = pd.read_csv("C:/Users/fmoss1/Downloads/Semester 3/Machine Learning/phishing_large.csv", header=None)

train_data_x, test_data_x, train_data_y, test_data_y = train_test_split(dataset.iloc[:, :30], dataset.iloc[:, -1:], test_size = 0.25)


lm = LinearRegression()
lm.fit(train_data_x, train_data_y)
linear_pred = lm.predict(test_data_x)
threshold = 0
linear_pred = (linear_pred > threshold).astype(int)
linear_pred[linear_pred == 0] = -1
mse = accuracy_score(test_data_y, linear_pred) * 100
print ("MSE Linear", mse)

plt.scatter(test_data_y, linear_pred)
plt.title("Prices vs Predicted prices")
plt.show()

ridge = Ridge()
ridge.fit(train_data_x, train_data_y)
ridge_pred = ridge.predict(test_data_x)
threshold = 0
ridge_pred = (ridge_pred > threshold).astype(int)
ridge_pred[ridge_pred == 0] = -1
mse = accuracy_score(test_data_y, ridge_pred) * 100
print ("MSE ridge", mse)

plt.scatter(test_data_y, ridge_pred)
plt.title("Prices vs Predicted prices")
plt.show()

lasso = Lasso()
lasso.fit(train_data_x, train_data_y)
lasso_pred = lasso.predict(test_data_x)
threshold = 0
lasso_pred = (lasso_pred > threshold).astype(int)
lasso_pred[lasso_pred == 0] = -1
mse = accuracy_score(test_data_y, lasso_pred) * 100
print ("MSE lasso", mse)

plt.scatter(test_data_y, lasso_pred)
plt.title("Prices vs Predicted prices")
plt.show()

logistic = LogisticRegression()
logistic.fit(train_data_x, train_data_y)
logistic_pred = logistic.predict(test_data_x)
threshold = 0
logistic_pred = (logistic_pred > threshold).astype(int)
logistic_pred[logistic_pred == 0] = -1
mse = accuracy_score(test_data_y, logistic_pred) * 100
print ("MSE logistic", mse)

plt.scatter(test_data_y, logistic_pred)
plt.title("Prices vs Predicted prices")
plt.show()

svm = svm.SVC()
svm.fit(train_data_x, train_data_y)
svm_pred = svm.predict(test_data_x)
threshold = 0
svm_pred = (svm_pred > threshold).astype(int)
svm_pred[svm_pred == 0] = -1
mse = accuracy_score(test_data_y, svm_pred) * 100
print ("MSE svm", mse)

plt.scatter(test_data_y, svm_pred)
plt.title("Prices vs Predicted prices")
plt.show()

nn = MLPClassifier()
nn.fit(train_data_x, train_data_y)
nn_pred = nn.predict(test_data_x)
threshold = 0
nn_pred = (nn_pred > threshold).astype(int)
nn_pred[nn_pred == 0] = -1
mse = accuracy_score(test_data_y, nn_pred) * 100
print ("MSE nn:", mse)

plt.scatter(test_data_y, nn_pred)
plt.title("Prices vs Predicted prices")
plt.show()

dt = tree.DecisionTreeClassifier()
dt.fit(train_data_x, train_data_y)
dt_pred = dt.predict(test_data_x)
threshold = 0
dt_pred = (dt_pred > threshold).astype(int)
dt_pred[dt_pred == 0] = -1
mse = accuracy_score(test_data_y, dt_pred) * 100
print ("MSE dt:", mse)

plt.scatter(test_data_y, dt_pred)
plt.title("Prices vs Predicted prices")
plt.show()

rf = RandomForestClassifier()
rf.fit(train_data_x, train_data_y)
rf_pred = rf.predict(test_data_x)
threshold = 0
rf_pred = (rf_pred > threshold).astype(int)
rf_pred[rf_pred == 0] = -1
mse = accuracy_score(test_data_y, rf_pred) * 100
print ("MSE rf:", mse)

plt.scatter(test_data_y, rf_pred)
plt.title("Prices vs Predicted prices")
plt.show()

kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(train_data_x, train_data_y)
kn_pred = kn.predict(test_data_x)
threshold = 0
kn_pred = (kn_pred > threshold).astype(int)
kn_pred[kn_pred == 0] = -1
mse = accuracy_score(test_data_y, kn_pred) * 100
print ("MSE kn:", mse)

plt.scatter(test_data_y, kn_pred)
plt.title("Prices vs Predicted prices")
plt.show()