#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:12:54 2017

@author: admin
"""
# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 


#import dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values   #: means all first is row second is column
Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# fit simple linear regression model to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting test results over obtained trained data
y_pred = regressor.predict(X_test)

#visualizing the training data results
plt.scatter(X_train,y_train,color = "red")
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
