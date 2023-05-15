#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 17:47:24 2023

@author: roatisiris
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

def put_in_bins(data, bins, way_to_bin):
    digitized = np.digitize(data,bins)
    
    if way_to_bin == 'binned_centre':
        midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
        new_data = midpoints_bins[digitized-1]
    elif way_to_bin == 'binned_random':
        bin_size = bins[1] - bins[0]
        random_points_bins = bins[:len(bins)-1] + random.uniform(0, bin_size)
        new_data = random_points_bins[digitized-1]
    elif way_to_bin == 'rank':
        new_data = digitized;
    return new_data

def calc_ss(x,y):
   return x.T @ y
   # return sum(x)

def generate_test(e, std, size, beta_0, beta_1):
    X = np.random.normal(e, std, size)
    y = beta_0 + beta_1 * X + np.random.normal(0, 1, size)
    return X, y

def mse(y_predicted, actual_y):
    return sum((y_predicted - actual_y)**2) / len(actual_y)

def transf(type_transf, list_wanted = None):
    if type_transf in ['binned_centre', 'binned_random','rank']:
        list_of_bins = []
        for i in range(len(list_wanted)):
            bin_size = list_wanted[i]
            bins = np.arange(bin_size/2,1000,bin_size)
            bins = np.concatenate((-bins[::-1], bins))
            list_of_bins.append(bins)
        return list_of_bins
    
def data_transf(X, type_transf, bins = None, constant = None):
    if type_transf in ['binned_centre', 'binned_random','rank']:
        return put_in_bins(X, bins, type_transf)
    elif type_transf == 'multiplied_non_random':
        return (1 + constant) * X
    
    
def predict_regression_cart(X, y, X_test, pruning = None,max_depth = None, min_samples_split = None, min_samples_leaf = None):
    if pruning == 'prepruning':
        regressor = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=  min_samples_split, min_samples_leaf = min_samples_leaf)
        regressor.fit(X.reshape(-1,1), y) #training the algorithm
    elif pruning == 'postpruning':
        regressor = post_pruning(X.reshape(-1,1),y)
    else:
        regressor = DecisionTreeRegressor()
        regressor.fit(X.reshape(-1,1), y) #training the algorithm

    # plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(regressor,filled=True)
    # plt.show()
    
    y_predicted = regressor.predict(X_test.reshape(-1,1))
 
    return y_predicted
    
def post_pruning(X,y):
    
    size_train = len(y)
    value_cut = 3 * size_train // 4
    
    
    X_train_tr = X[: value_cut]
    X_train_test = X[value_cut:]
    y_train_tr = y[: value_cut]
    y_train_test = y[value_cut:]
    
    clf = DecisionTreeRegressor()
    
    path = clf.cost_complexity_pruning_path(X_train_tr, y_train_tr)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeRegressor(ccp_alpha=ccp_alpha)
        clf.fit(X_train_tr, y_train_tr)
        clfs.append(clf)
    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]
    
    train_mse = []
    test_mse = []
    
    
    for c in clfs:

        y_train_tr_pred = c.predict(X_train_tr)
        y_train_test_pred = c.predict(X_train_test)
        
        # print(y_train_tr_pred.shape)
        # print(y_train_test_pred.shape)
        # print(y_train_tr.shape)
        # print(y_train_test.shape)
        
      
        
        
        # print(y_train_tr_pred)
        # print(y_train_test_pred)
        
        
        # stop
        
        print(mse(y_train_tr_pred, y_train_tr))
        print(mse(y_train_test_pred, y_train_test))
        
        train_mse.append((mse(y_train_tr_pred, y_train_tr)))
      
        test_mse.append((mse(y_train_test_pred, y_train_test)))
      

    min_difference = abs(train_mse[0] - test_mse[0])
    min_difference_index = 0
    
    for i in range(1, len(train_mse)):
        diff = abs(train_mse[i] - test_mse[i])
        if diff < min_difference:
            min_difference = diff
            min_difference_index = i
    
    chosen_model = clfs[min_difference_index]
    
    return chosen_model
    
    
    
    
        
    
def ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra=None, pruning = None, max_depth = None, min_samples_split = None, min_samples_leaf = None):
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]
    
    how_many_extras = len(extra) + 1

    abs_diff_ss_dictionary = {'none':np.zeros((how_many_extras, how_many_it))}
    mse_testdata_dictionary = {'none':np.zeros((how_many_extras, how_many_it))}
        

    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)


    for iteration in range(how_many_it):
        if iteration % 20 == 0:
            print(np.round((iteration+1)/how_many_it,2))    # code progress
        
        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
        
       
        
        y_predicted_unbinned = predict_regression_cart(X, y, X_test, pruning = pruning, max_depth=max_depth, min_samples_split=  min_samples_split, min_samples_leaf = min_samples_leaf)
        mse_unbinned = mse(y_predicted_unbinned, y_test)
        ss_unbinned = calc_ss(X, y)
        diff = ss_unbinned - ss_unbinned
        abs_diff_ss_dictionary['none'] [0, iteration] = diff**2
        mse_testdata_dictionary['none'] [0, iteration] =  mse_unbinned
       
        if type_transf in ['binned_centre', 'binned_random', 'rank']:
            list_of_bins = transf(type_transf, list_wanted = extra)
            
        for i in range(how_many_extras - 1):
            
            print('ITERATION'+str(i))
            
            if type_transf in ['binned_centre', 'binned_random','rank']:
                bins = list_of_bins[i] 
                new_X = data_transf(X, type_transf, bins)
                
            elif type_transf == 'multiplied_non_random':
                new_X = data_transf(X, type_transf, constant = extra[i])
            
            y_predicted_binned = predict_regression_cart(new_X, y, X_test,pruning = pruning, max_depth=max_depth, min_samples_split=  min_samples_split, min_samples_leaf = min_samples_leaf)
            
            mse_binned = mse(y_predicted_binned, y_test)
            ss_binned = calc_ss(new_X, y)
            diff = ss_unbinned - ss_binned
            abs_diff_ss_dictionary['none'] [i+1, iteration] = diff**2
            mse_testdata_dictionary['none'] [i+1, iteration] =  mse_binned
            
           
    return abs_diff_ss_dictionary, mse_testdata_dictionary
   


how_many_it = 1
size_test, size_train = 1000, 100
parameters = [1, 1, 0, 1];
# type_transf = 'multiplied_non_random'
type_transf = 'binned_centre'


if type_transf in ['binned_centre', 'binned_random','rank']:
    max_bin_size = 15
    extra = np.linspace(0.01, max_bin_size, 10)
elif type_transf == 'multiplied_non_random':
    extra = np.linspace(0.01, 5, 50)

# abs_diff_ss_dictionary, mse_testdata_dictionary = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra=extra, pruning = 'postpruning')


#### SOME PRE-PRUNING

string_title = 'Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])+', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test)+'\n Transformation: cart'

list_max_depth = [5,7]
list_min_samples_split=[2, 4]
list_min_samples_leaf=[2, 3]

plt.figure()
for max_depth in list_max_depth:
    for min_samples_split in list_min_samples_split:
        for min_samples_leaf in list_min_samples_leaf:
            abs_diff_ss_dictionary, mse_testdata_dictionary = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra, max_depth=max_depth, min_samples_split=  min_samples_split, min_samples_leaf = min_samples_leaf)
            plt.plot(np.mean(abs_diff_ss_dictionary['none'], axis = 1),np.mean(mse_testdata_dictionary['none'] , axis = 1),'.', label = str(max_depth) +',' +str(min_samples_split) +','+str(min_samples_leaf) )
            
            
plt.legend()
plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
plt.ylabel('predictive MSE')
plt.title(string_title)
plt.show()

    
