#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 11:59:58 2023

@author: roatisiris
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data

def calc_ss(x,y):
    return x.T @ y

def generate_test(e1, std1, e2, std2, size1, size2):
    s1 = np.random.normal(e1, std1, size1)
    s2 = np.random.normal(e2, std2, size2)
    
    X = np.concatenate((s1.reshape((size1,)),s2.reshape((size2,))))
    y = np.concatenate((0*np.ones((size1,)),np.ones((size2,))))
    return X, y

def compute_accuracy(y_predicted, actual_y):
    coincide = 0
    n = len(y_predicted)
    for i in range(n):
        if y_predicted[i] == actual_y[i]:
            coincide += 1
    return coincide/n

size_train1 = 200
size_train2 = 100
size_test1 = 100
size_test2 = 100

e1 = 0
e2 = 2
std1 = 1
std2 = 1

list_bin_sizes = np.linspace(0.01, 20, 100)

how_many = 100

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


for iteration in range(how_many):    
    print((iteration+1)/how_many)
    X,y = generate_test(e1, std1, e2, std2, size_train1, size_train2)
    X_test, y_test = generate_test(e1, std1, e2, std2, size_test1, size_test2)

    regressor = LogisticRegression(penalty='none')
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    y_predicted_unbinned = regressor.predict(X_test.reshape(-1,1))
    mse_unbinned = compute_accuracy(y_predicted_unbinned, y_test)
    
    ss_unbinned = calc_ss(X, y)
    
    diff = ss_unbinned - ss_unbinned
    difference_ss[0][iteration],  abs_diff_ss[0][iteration] = diff, abs(diff)**2
    mse_testdata[0][iteration] = mse_unbinned
    
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(X, bins)
      
        regressor = LogisticRegression(penalty='none')
        regressor.fit(new_X.reshape(-1,1), y) #training the algorithm
        y_predicted_binned = regressor.predict(X_test.reshape(-1,1))
        mse_binned = compute_accuracy(y_predicted_binned, y_test)
        
        ss_binned = calc_ss(new_X, y)
        
        diff = ss_unbinned - ss_binned
        difference_ss[i+1][iteration],  abs_diff_ss[i+1][iteration] = diff, abs(diff)**2
   
        mse_testdata[i+1][iteration] = mse_binned


### fitting a second order polynomial



plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
plt.xlabel('mean abs(ss_unbinned - ss_binned)')
plt.ylabel('mean accuracy') 
plt.title('Logistic')
plt.show()

plt.scatter([0] + list(list_bin_sizes), np.mean(abs_diff_ss,axis = 1))
# plt.plot(polyline, model(polyline), 'r', label = string_model)
# plt.legend()
plt.ylabel('mean abs(ss_unbinned - ss_binned)^2')
plt.xlabel('bin_size')
# plt.title('Linear - Polynomial Fit of Order ' +str(len(coeff)-1))
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
