#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 11:16:51 2023

@author: roatisiris
"""

import pandas as pd
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing



def dataset_to_Xandy(dataset):
    X = dataset.loc[:, dataset.columns != 'is_claim']
    y = dataset.loc[:, dataset.columns == 'is_claim']
    return X, y

def split_dataset(X,y):
    size = X.shape[0]
    stop = 2 * size//3
    X_train = X.iloc[:stop,:]
    y_train = y.iloc[:stop,:]
    X_test = X.iloc[stop:,:]
    y_test = y.iloc[stop:,:]
    return X_train, y_train, X_test, y_test

df = pd.read_csv('cardataset_train.csv')
# print(df.age_of_car.value_counts())
# unique_ages = df.age_of_car.unique()
# unique_engine_type = df.engine_type.unique()




### SIMPLE ENCODING

new_df = df[['policy_tenure','age_of_car','engine_type','is_claim']]

le = preprocessing.LabelEncoder()
le.fit(new_df["engine_type"])
new_df["engine_type"] = le.transform(new_df["engine_type"])

X,y =  dataset_to_Xandy(new_df)

# split into train test sets
X_train, y_train, X_test, y_test = split_dataset(X,y)

logistic_regressor = LogisticRegression(penalty = 'none')  # create object for the class
logistic_regressor.fit(X_train, y_train)  # perform linear regression
y_predicted = logistic_regressor.predict(X_test)

print(logistic_regressor.coef_, logistic_regressor.intercept_)

confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.title('Confusion matrix for simple encoding')
plt.show()




### ONE HOT ENCODING

new_df = df[['policy_tenure','age_of_car','engine_type','is_claim']]
encoder = ce.OneHotEncoder(cols='engine_type',return_df=True,use_cat_names=True)

data_encoded = encoder.fit_transform(new_df['engine_type'])

new_df = new_df.drop(['engine_type'], axis=1)
new_df = pd.concat([new_df, data_encoded], axis=1)


X,y =  dataset_to_Xandy(new_df)

# split into train test sets
X_train, y_train, X_test, y_test = split_dataset(X,y)

logistic_regressor = LogisticRegression(penalty = 'none')  # create object for the class
logistic_regressor.fit(X_train, y_train)  # perform linear regression
y_predicted = logistic_regressor.predict(X_test)

print(logistic_regressor.coef_, logistic_regressor.intercept_)

confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.title('Confusion matrix for one hot encoding')
plt.show()


##### TARGET ENCODING
new_df = df[['policy_tenure','age_of_car','engine_type','is_claim']]
tenc=ce.TargetEncoder() 
df_engine_type = tenc.fit_transform(new_df['engine_type'],new_df['is_claim'])

new_df = df_engine_type.join(new_df.drop('engine_type',axis = 1))
X,y =  dataset_to_Xandy(new_df)
X_train, y_train, X_test, y_test = split_dataset(X,y)

logistic_regressor = LogisticRegression(penalty = 'none')  # create object for the class
logistic_regressor.fit(X_train, y_train)  # perform linear regression
y_predicted = logistic_regressor.predict(X_test)

print(logistic_regressor.coef_, logistic_regressor.intercept_)

confusion_matrix = metrics.confusion_matrix(y_test, y_predicted)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
cm_display.plot()
plt.title('Confusion matrix for target encoding')
plt.show()





