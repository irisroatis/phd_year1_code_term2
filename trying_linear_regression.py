#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 15:26:18 2022

@author: ir318
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data

beta_0_true = 0.4
beta_1_true = 0.6

size = 100
list_bin_sizes = [0.01,0.05,0.1,0.5, 0.75, 1, 1.5, 2, 3, 4]

how_many = 1000

list_error_beta0 = [np.zeros((how_many,))]
list_error_beta1 = [np.zeros((how_many,))]
corrected_grad = [np.zeros((how_many,))]
residual_variance = [np.zeros((how_many,))]
residual_variance_corrected = [np.zeros((how_many,))]

list_of_bins = []
for i in range(len(list_bin_sizes)):
    list_error_beta0.append(np.zeros((how_many,)))
    list_error_beta1.append(np.zeros((how_many,)))
    corrected_grad.append(np.zeros((how_many,)))
    residual_variance.append(np.zeros((how_many,)))
    residual_variance_corrected.append(np.zeros((how_many,)))
    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,how_many,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    list_of_bins.append(bins)
    

for iteration in range(how_many):
    X = np.random.normal(5, 5, size)
    y = beta_0_true + beta_1_true * X + np.random.normal(0, 1, size)
    
    regressor = LinearRegression()  
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    list_error_beta0[0][iteration] = regressor.intercept_ - beta_0_true
    list_error_beta1[0][iteration] = regressor.coef_[0] - beta_1_true
    residual_variance[0][iteration] = sum((y - regressor.intercept_ - regressor.coef_[0]*X)**2) / (size-2)
    residual_variance_corrected[0][iteration] =   residual_variance[0][iteration] 
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(X, bins)
        variance_sample = np.var(new_X)
        regressor = LinearRegression()  
        regressor.fit(new_X.reshape(-1,1), y) #training the algorithm
        list_error_beta0[i+1][iteration] = regressor.intercept_ - beta_0_true
        list_error_beta1[i+1][iteration] = regressor.coef_[0] - beta_1_true
        correct =  regressor.coef_[0] / (1 - list_bin_sizes[i]**2 / (12 * variance_sample))
        corrected_grad[i+1][iteration] = correct - beta_1_true
        residual_variance[i+1][iteration] = sum((y - regressor.intercept_ - regressor.coef_[0]*new_X)**2) / (size-2)
        residual_variance_corrected[i+1][iteration] = sum((y - regressor.intercept_ - correct * new_X)**2) / (size-2)


dictionary_beta0 = {'none':list_error_beta0[0]}
dictionary_beta1 = {'none':list_error_beta1[0]}
dictionary_corrected_grad = {}
dictionary_res = {'none':residual_variance[0]}
dictionary_res_corrected = {'none':residual_variance_corrected[0]}
for i in range(len(list_bin_sizes)):
    dictionary_beta0[str(list_bin_sizes[i])] = list_error_beta0[i+1]
    dictionary_beta1[str(list_bin_sizes[i])] = list_error_beta1[i+1]
    dictionary_corrected_grad[str(list_bin_sizes[i])] = corrected_grad[i+1]
    dictionary_res[str(list_bin_sizes[i])] = residual_variance[i+1]
    dictionary_res_corrected[str(list_bin_sizes[i])] = residual_variance_corrected[i+1]
    
# Pandas dataframe
data0 = pd.DataFrame(dictionary_beta0)
# Plot the dataframe
ax = data0[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2', '3', '4']].plot(kind='box', title='boxplot')
# Display the plot
plt.show()

# Pandas dataframe
data1 = pd.DataFrame(dictionary_beta1)
# Plot the dataframe
ax = data1[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2', '3', '4']].plot(kind='box', title='boxplot')
# Display the plot
plt.show()

# Pandas dataframe
data2 = pd.DataFrame(dictionary_corrected_grad)
# Plot the dataframe
ax = data2[['0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2', '3', '4']].plot(kind='box', title='boxplot')
# Display the plot
plt.show()

# Pandas dataframe
data3 = pd.DataFrame(dictionary_res)
# Plot the dataframe
ax = data3[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2', '3', '4']].plot(kind='box', title='boxplot')
# Display the plot
plt.show()

# Pandas dataframe
data4 = pd.DataFrame(dictionary_res_corrected)
# Plot the dataframe
ax = data4[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2', '3', '4']].plot(kind='box', title='boxplot')
# Display the plot
plt.show()
      
      
            
