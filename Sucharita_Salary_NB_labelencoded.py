# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 13:10:24 2020

@author: SUCHARITA
"""
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB 
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# we have to classify teh salary data
s_train = pd.read_csv("F:\\ExcelR\\Assignment\\NB\\SalaryData_Train.csv")
s_test = pd.read_csv("F:\\ExcelR\\Assignment\\NB\\SalaryData_Test.csv")
s_train.shape
s_train.info() # 5 numeric and 9 categorical variables
s_train['Salary'].unique()
s_train.head()
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

# train data analysis

categorical = [var for var in s_train.columns if s_train[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
s_train[categorical].isnull().sum() # shows no null values

for var in categorical:  # view frequency of each categorical variable type, no junk values are there
    print(s_train[var].value_counts())
    
for var in categorical: 
    print(s_train[var].value_counts()/np.float(len(s_train)))    # all proportions add upto 100%

for var in categorical:  # check different unique lebels for each
    print(var, ' contains ', len(s_train[var].unique()), ' labels')
    
# check if any column have junk values, do it for all categories
s_train.workclass.unique() # 7 classes
s_train.workclass.value_counts() # give frequency of each class

numerical = [var for var in s_train.columns if s_train[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

for var in numerical:  # view frequency of each numerical variable type, no junk values are there
    print(s_train[var].value_counts())
    
for var in numerical: 
    print(s_train[var].value_counts()/np.float(len(s_train)))    # all proportions add upto 100%

for var in numerical:  # check different unique lebels for each
    print(var, ' contains ', len(s_train[var].unique()), ' labels')

# check if any column have junk values, do it for all categories
s_train.capitalgain.unique() # type of values, no junk values
s_train.capitalgain.value_counts() # give frequency of each value


# test data analysis

categorical = [var for var in s_test.columns if s_test[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)
s_test[categorical].isnull().sum() # shows no null values

for var in categorical:  # view frequency of each categorical variable type, no junk values are there
    print(s_test[var].value_counts())
    
for var in categorical: 
    print(s_test[var].value_counts()/np.float(len(s_test)))    # all proportions add upto 100%

for var in categorical:  # check different unique lebels for each
    print(var, ' contains ', len(s_test[var].unique()), ' labels')
    
# check if any column have junk values, do it for all categories
s_test.workclass.unique() # 7 classes
s_test.workclass.value_counts() # give frequency of each class

numerical = [var for var in s_test.columns if s_test[var].dtype!='O']
print('There are {} numerical variables\n'.format(len(numerical)))
print('The numerical variables are :', numerical)

for var in numerical:  # view frequency of each numerical variable type, no junk values are there
    print(s_test[var].value_counts())
    
for var in numerical: 
    print(s_test[var].value_counts()/np.float(len(s_train)))    # all proportions add upto 100%

for var in numerical:  # check different unique lebels for each
    print(var, ' contains ', len(s_test[var].unique()), ' labels')

# check if any column have junk values, do it for all categories
s_test.capitalgain.unique() # type of values, no junk values
s_test.capitalgain.value_counts()

# Model building and evaluation

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
   s_train[i] = number.fit_transform(s_train[i])
   s_test[i] = number.fit_transform(s_test[i])

colnames = s_train.columns
len(colnames[0:13])
x_train = s_train[colnames[0:13]]
y_train = s_train[colnames[13:14]]
x_test  = s_test[colnames[0:13]]
y_test  = s_test[colnames[13:14]]

gnb = GaussianNB()
mnb = MultinomialNB()

pred_gnb = gnb.fit(x_train,y_train).predict(x_test)
confusion_matrix(y_test,pred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 79.4%
pd.crosstab(y_test.values.flatten(),pred_gnb) # confusion matrix using 
np.mean(pred_gnb==y_test.values.flatten()) # 79.4%


pred_mnb = mnb.fit(x_train,y_train).predict(x_test)
confusion_matrix(y_test,pred_mnb)
print("Accuracy",(10891+780)/(10891+469+2920+780))  # 77.5%
pd.crosstab(y_test.values.flatten(),pred_mnb) # confusion matrix using 
np.mean(pred_mnb==y_test.values.flatten()) # 77.5%
