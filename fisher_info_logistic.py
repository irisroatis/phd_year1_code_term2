#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 17:18:52 2022

@author: ir318
"""

import numpy as np
import matplotlib.pyplot as plt
from defined_functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error


size = 500

mu1 = np.array([0.25])
mu2 = np.array([1])
sigma1 = np.array([1])
sigma2 = np.array([1])

def fisher_matrix(X,y):
    regressor = LogisticRegression()  
    regressor.fit(X, y) #training the algorithm
    
    beta0hat = regressor.intercept_ 
    beta1hat = regressor.coef_[0] 
    
    helper = np.exp(beta0hat + X * beta1hat)
    vector = np.divide(helper,(1+helper)**2)
    vector2 = np.multiply(X, vector)
    
    I = np.zeros((2,2))
    I[0,0] = sum(vector)
    I[1,0] = I[0,1] = sum(vector2)
    I[1,1] = sum( np.multiply(X, vector2))

    return I

def inverse_2by2(matrix):
    determinant = matrix[0,0] * matrix[1,1] - matrix[1,0] * matrix[0,1]
    inverse = np.zeros_like(matrix)
    inverse[0,0] = matrix[1,1]
    inverse[1,1] = matrix[0,0]
    inverse[0,1] = -matrix[0,1]
    inverse[1,0] = -matrix[1,0]
    inverse /= determinant
    return inverse


def inv_fisher_normalised(X,y):
    inv_fisher = inverse_2by2(fisher_matrix(X,y))
    helper = np.sqrt(inv_fisher[0,0]*inv_fisher[1,1]) 
    inv_fisher[0,1] /= helper
    inv_fisher[1,0] /= helper
    return inv_fisher

list_bin_sizes = [0.01, 0.05,0.1,0.5, 0.75, 1, 1.5, 2, 3]

how_many = 400

element00 = [np.zeros((how_many,))]
element10 = [np.zeros((how_many,))]
element11 = [np.zeros((how_many,))]

list_of_bins = []
for i in range(len(list_bin_sizes)):
    element00.append(np.zeros((how_many,)))
    element10.append(np.zeros((how_many,)))
    element11.append(np.zeros((how_many,)))
    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,how_many,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    list_of_bins.append(bins)
    
y = np.concatenate((0*np.ones((size,)),np.ones((size,))))

for iterations in range(how_many):
     s1 = np.random.normal(mu1, sigma1, size)
     s2 = np.random.normal(mu2, sigma2, size)
    
     # create X and y to apply classification
     X = np.concatenate((s1.reshape((size,1)),s2.reshape((size,1))))
     I_unbinned = inv_fisher_normalised(X, y)
     
     element00[0][iterations] = I_unbinned[0,0]
     element10[0][iterations] = I_unbinned[1,0]
     element11[0][iterations] = I_unbinned[1,1]
     
     for i in range(len(list_bin_sizes)):
         bins = list_of_bins[i] 
         X_binned = put_in_bins(X, bins)
         I_binned = inv_fisher_normalised(X_binned,y)
         
         element00[i+1][iterations] = I_binned[0,0]
         element10[i+1][iterations] = I_binned[1,0]
         element11[i+1][iterations] = I_binned[1,1]

dictionary_element00 = {'none':element00[0]}
dictionary_element10 = {'none':element10[0]}
dictionary_element11 = {'none':element11[0]}
for i in range(len(list_bin_sizes)):
    dictionary_element00[str(list_bin_sizes[i])] = element00[i+1] 
    dictionary_element10[str(list_bin_sizes[i])] = element10[i+1] 
    dictionary_element11[str(list_bin_sizes[i])] = element11[i+1]
    
    
# Pandas dataframe
data0 = pd.DataFrame(dictionary_element00)
# Plot the dataframe
ax = data0[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2']].plot(kind='box', title='boxplot')
# Display the plot
plt.title('$C_{11}$ logistic')
plt.show()

# Pandas dataframe
data0 = pd.DataFrame(dictionary_element10)
# Plot the dataframe
ax = data0[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2']].plot(kind='box', title='boxplot')
# Display the plot
plt.title('$C_{12}$ logistic')

plt.show()
    
# Pandas dataframe
data0 = pd.DataFrame(dictionary_element11)
# Plot the dataframe
ax = data0[['none','0.01','0.05','0.1','0.5', '0.75', '1', '1.5', '2']].plot(kind='box', title='boxplot')
# Display the plot
plt.title('$C_{22}$ logistic')

plt.show()