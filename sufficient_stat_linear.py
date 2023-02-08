#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:45:13 2023

@author: roatisiris
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas as pd

def logistic(x):
    logistic =  1/(1+np.exp(-x))
    return logistic

def optimise(X, y, num_iterations, learning_rate = 0.005):
    n = X.shape[0] 
    beta = [0]
    beta_0 = [0]
    for i in range(num_iterations):
        # Calculate gradients 
        
        y_log = logistic(X @ beta + beta_0) 
     
        dbeta = (1/n) * np.transpose(X) @ (y_log - y)
        dbeta_0 = np.mean(y_log-y)
        
        # Updating procedure for the parameters (gradient descent)
        beta -= learning_rate * dbeta
        beta_0 -= learning_rate * dbeta_0 
 
    # Save parameters and gradients in dictionary
    params = {"beta": beta, "beta_0": beta_0}
    return params


def put_in_bins(data, bins, way_to_bin):
    digitized = np.digitize(data,bins)
    
    if way_to_bin == 'binned_centre':
        midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
        new_data = midpoints_bins[digitized-1]
    elif way_to_bin == 'binned_random':
        bin_size = bins[1] - bins[0]
        random_points_bins = bins[:len(bins)-1] + random.uniform(0, bin_size)
        new_data = random_points_bins[digitized-1]
    return new_data

def calc_ss(x,y):
    return x.T @ y

def generate_test(e, std, size, beta_0, beta_1):
    X = np.random.normal(e, std, size)
    y = beta_0 + beta_1 * X + np.random.normal(0, 1, size)
    return X, y

def mse(y_predicted, actual_y):
    return sum((y_predicted - actual_y)**2) / len(actual_y)

def transf(type_transf, list_wanted = None):
    if type_transf in ['binned_centre', 'binned_random']:
        list_of_bins = []
        for i in range(len(list_wanted)):
            bin_size = list_wanted[i]
            bins = np.arange(bin_size/2,1000,bin_size)
            bins = np.concatenate((-bins[::-1], bins))
            list_of_bins.append(bins)
        return list_of_bins
    
def data_transf(X, type_transf, bins = None, constant = None):
    if type_transf in ['binned_centre', 'binned_random']:
        return put_in_bins(X, bins, type_transf)
    elif type_transf == 'multiplied_non_random':
        return (1 + constant) * X
    
def ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra=None):
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]

    how_many_extras = len(extra) + 1


    difference_ss = np.zeros((how_many_extras, how_many_it))
    abs_diff_ss =  np.zeros((how_many_extras, how_many_it))
    mse_testdata =  np.zeros((how_many_extras, how_many_it))
     
    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)

    
    for iteration in range(how_many_it):
        
        print((iteration+1)/how_many_it)    # code progress
        
        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
    
        regressor = LinearRegression()  
        regressor.fit(X.reshape(-1,1), y) #training the algorithm
        y_predicted_unbinned = regressor.predict(X_test.reshape(-1,1))
        mse_unbinned = mse(y_predicted_unbinned, y_test)
        
        ss_unbinned = calc_ss(X, y)
        
        diff = ss_unbinned - ss_unbinned
        difference_ss[0, iteration],  abs_diff_ss[0,iteration] = diff, diff**2
        mse_testdata[0, iteration] = mse_unbinned
        
        if type_transf in ['binned_centre', 'binned_random']:
            list_of_bins = transf(type_transf, list_wanted = extra)
            
        for i in range(how_many_extras - 1):
            
            if type_transf in ['binned_centre', 'binned_random']:
                bins = list_of_bins[i] 
                new_X = data_transf(X, type_transf, bins)
                
            elif type_transf == 'multiplied_non_random':
                new_X = data_transf(X, type_transf, constant = extra[i])
                
            regressor = LinearRegression()  
            regressor.fit(new_X.reshape(-1,1), y) #training the algorithm
            y_predicted_binned = regressor.predict(X_test.reshape(-1,1))
            mse_binned = mse(y_predicted_binned, y_test)
        
            ss_binned = calc_ss(new_X, y)
            
            diff = ss_unbinned - ss_binned
            difference_ss[i+1,iteration],  abs_diff_ss[i+1,iteration] = diff, diff**2
       
            mse_testdata[i+1,iteration] = mse_binned
    
    return difference_ss, abs_diff_ss, mse_testdata

def plotting_against_mse(abs_diff_ss, mse_testdata, type_transf, parameters, size_test, size_train, how_many_it):
    plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
    plt.xlabel('$E[(S(D) - S(T(D)))^2]$')
    plt.ylabel('MSE')
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
              +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
              +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
    plt.show()
    
def plotting_width_against_ss(abs_diff_ss, extra, type_transf, parameters, size_test, size_train, how_many_it, line_of_fit = False):
    plt.scatter([0] + list(extra), np.mean(abs_diff_ss,axis = 1))
    plt.ylabel('$E[(S(D) - S(T(D)))^2]$')
    plt.xlabel('bin size')
    
    
    if line_of_fit:
        log_X = np.array([0] + list(extra)).reshape(-1,1)
        log_y = np.mean(abs_diff_ss,axis = 1).reshape(-1,1)
        
        scaled_log_y = log_y / max(log_y)
        
        final_model = optimise(log_X,scaled_log_y, num_iterations = 100, learning_rate=0.001)
        
        random_x = np.linspace(0, max(log_X),100)
        random_y = logistic(random_x @ final_model['beta_0'] + final_model['beta'][0,-1])
        plt.plot(random_x, random_y *  max(log_y), 'r',label='best fit')
        plt.legend()
        
    
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
              +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
              +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
    
    plt.show()


how_many_it = 200
size_test, size_train = 800, 100
parameters = [0.4, 1, 0, 1];
# type_transf = 'multiplied_non_random'
type_transf = 'binned_centre'
random.seed(1)

if type_transf in ['binned_centre', 'binned_random']:
    max_bin_size = 10
    extra = np.linspace(0.01, max_bin_size, 200)
elif type_transf == 'multiplied_non_random':
    extra = np.linspace(0.01, 5, 50)

difference_ss, abs_diff_ss, mse_testdata = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra)
plotting_against_mse(abs_diff_ss, mse_testdata, type_transf, parameters, size_test, size_train, how_many_it)
plotting_width_against_ss(abs_diff_ss, extra, type_transf, parameters, size_test, size_train, how_many_it, True)

#### fitting logistic regression for bin size against E(sq difference)
