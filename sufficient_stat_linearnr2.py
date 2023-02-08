#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:45:13 2023

@author: roatisiris
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

def calc_ss(x,y):
    return x.T @ y

def generate_test(e, std, size, beta_0, beta_1):
    X = np.random.normal(e, std, size)
    y = beta_0 + beta_1 * X + np.random.normal(0, 1, size)
    return X, y

def mse(y_predicted, actual_y):
    return sum((y_predicted - actual_y)**2) / len(actual_y)

beta_0_true = 0.4
beta_1_true = 1
e = 0
std = 1

size_train = 400
size_test = 1000
max_bin_size = 5
list_bin_sizes = np.linspace(0.01, max_bin_size, 50)

how_many = 800

difference_ss = [np.zeros((how_many,))]
abs_diff_ss =  [np.zeros((how_many,))]
mse_testdata =  [np.zeros((how_many,))]

list_of_bins = []
for i in range(len(list_bin_sizes)):
 
    difference_ss.append(np.zeros((how_many,)))
    abs_diff_ss.append(np.zeros((how_many,)))
    mse_testdata.append(np.zeros((how_many,)))

    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,1000,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    list_of_bins.append(bins)
    

X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)


for iteration in range(how_many):
    print((iteration+1)/how_many)    
    X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)

    regressor = LinearRegression()  
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    y_predicted_unbinned = regressor.predict(X_test.reshape(-1,1))
    mse_unbinned = mse(y_predicted_unbinned, y_test)
    
    ss_unbinned = calc_ss(X, y)
    
    diff = ss_unbinned - ss_unbinned
    difference_ss[0][iteration],  abs_diff_ss[0][iteration] = diff, abs(diff)**2
    mse_testdata[0][iteration] = mse_unbinned
    
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(X, bins)
      
        regressor = LinearRegression()  
        regressor.fit(new_X.reshape(-1,1), y) #training the algorithm
        y_predicted_binned = regressor.predict(X_test.reshape(-1,1))
        mse_binned = mse(y_predicted_binned, y_test)
        
        ss_binned = calc_ss(new_X, y)
        
        diff = ss_unbinned - ss_binned
        difference_ss[i+1][iteration],  abs_diff_ss[i+1][iteration] = diff, abs(diff)**2
   
        mse_testdata[i+1][iteration] = mse_binned


### fitting a second order polynomial

# coeff = np.polyfit(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1), 2)
# model = np.poly1d(coeff)
# polyline = np.linspace(0, max(np.mean(abs_diff_ss,axis = 1)), 50)

# string_model = ''
# for i in range(len(coeff)):
#     if i != len(coeff) - 1: 
#         string_model += str(np.round(coeff[i],5)) + 'x^' +str(len(coeff) - 1- i) + '+ '
#     else:
#         string_model += str(np.round(coeff[i],5)) 


plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
# plt.plot(polyline, model(polyline), 'r', label = string_model)
# plt.legend()
plt.xlabel('mean abs(ss_unbinned - ss_binned)^2')
plt.ylabel('mean of MSE')
plt.title('Linear - E(|S(D) - S(D*)|^2) against MSE')
plt.show()

plt.scatter([0] + list(list_bin_sizes), np.mean(abs_diff_ss,axis = 1))
# plt.plot(polyline, model(polyline), 'r', label = string_model)
# plt.legend()
plt.ylabel('mean abs(ss_unbinned - ss_binned)^2')
plt.xlabel('bin_size')
plt.title('Linear - bin_size against E(|S(D) - S(D*)|^2)')
plt.show()

# ### fitting an exponential polynomial
# p = np.polyfit(np.mean(abs_diff_ss,axis = 1), np.log(np.mean(mse_testdata,axis = 1)), 1)
# a = np.exp(p[1])
# b = p[0]
# polyline = np.linspace(0, max(np.mean(abs_diff_ss,axis = 1)), 50)
# y_fitted = a * np.exp(b * polyline)

# string_model = str(np.round(a,4)) + '* e^(' +str(np.round(b,4)) +'x)'

# plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
# plt.plot(polyline, y_fitted, 'r', label = string_model)
# plt.legend()
# plt.xlabel('abs(ss_unbinned - ss_binned)')
# plt.ylabel('MSE')
# plt.title('Exponential Fit')
# plt.show()


# plt.scatter(np.mean(difference_ss,axis = 1), np.mean(mse_testdata,axis = 1))
# plt.legend()
# plt.xlabel('ss_unbinned - ss_binned')
# plt.ylabel('MSE')
# plt.title('Difference of SS')
# plt.show()
