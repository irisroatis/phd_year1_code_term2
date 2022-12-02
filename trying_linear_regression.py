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

list_of_bins = []
for i in range(len(list_bin_sizes)):
    list_error_beta0.append(np.zeros((how_many,)))
    list_error_beta1.append(np.zeros((how_many,)))
    corrected_grad.append(np.zeros((how_many,)))
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
    
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(X, bins)
        variance_sample = np.var(new_X)
        regressor = LinearRegression()  
        regressor.fit(new_X.reshape(-1,1), y) #training the algorithm
        list_error_beta0[i+1][iteration] = regressor.intercept_ - beta_0_true
        list_error_beta1[i+1][iteration] = regressor.coef_[0] - beta_1_true
        corrected_grad[i+1][iteration] = regressor.coef_[0] / (1 - list_bin_sizes[i]**2 / (12 * variance_sample))
        
dictionary_beta0 = {'none':list_error_beta0[0]}
dictionary_beta1 = {'none':list_error_beta1[0]}
dictionary_corrected_grad = {}
for i in range(len(list_bin_sizes)):
    dictionary_beta0[str(list_bin_sizes[i])] = list_error_beta0[i+1]
    dictionary_beta1[str(list_bin_sizes[i])] = list_error_beta1[i+1]
    dictionary_corrected_grad[str(list_bin_sizes[i])] = corrected_grad[i+1]
    
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

            
            
