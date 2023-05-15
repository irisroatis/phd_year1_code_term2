#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 13:45:13 2023

@author: roatisiris
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate
from sklearn import tree

def logistic_curve(X, a, b, c, d):
    """
    Logistic function with parameters a, b, c, d
    a is the curve's maximum value (top asymptote)
    b is the curve's minimum value (bottom asymptote)
    c is the logistic growth rate or steepness of the curve
    d is the x value of the sigmoid's midpoint
    """
    return (a / (1 + np.exp(-c * (X - d)))) + b

def optimise_logistic(X, y):
    p0 = [max(y), 1, 1, 5]
    logistic_params, _ = curve_fit(logistic_curve, X, y, p0 = p0)
    return logistic_params

def quadratic(X, a, b, c):
    return a* X **2 + b*X + c

def optimise_quadratic(X,y):
    q_param,_ = curve_fit(quadratic, X, y)
    return q_param

def root_function(x, a, b):
    return x**(1/a) * b

def optimise_root(X, y):
    root_params, _ = curve_fit(root_function, X, y)
    return root_params

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
            # bins = np.arange(0,1000,bin_size)
            # bins = np.concatenate((-bins[::-1][:-1], bins))
            
            bins = np.arange(bin_size/2,1000,bin_size)
            bins = np.concatenate((-bins[::-1], bins))
            
            list_of_bins.append(bins)
        return list_of_bins
    
def data_transf(X, type_transf, bins = None, constant = None):
    if type_transf in ['binned_centre', 'binned_random','rank']:
        return put_in_bins(X, bins, type_transf)
    elif type_transf == 'multiplied_non_random':
        return (1 + constant) * X
    
    
def predict_regressions(X, y, X_test, type_regression, a = None):
    if type_regression =='ridge':
        regressor = Ridge(alpha = a)  
    elif type_regression == 'lasso':
        regressor = Lasso(alpha = a)
    elif type_regression == 'simple':
        regressor = LinearRegression()      
    elif type_regression == 'cart':
        regressor = DecisionTreeRegressor()  
        
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    
    y_predicted = regressor.predict(X_test.reshape(-1,1))
    
    return y_predicted
    
def ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra=None, type_regression = None, alpha = None):
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]
    
    how_many_extras = len(extra) + 1
    

    abs_diff_ss_dictionary = {'none':np.zeros((how_many_extras, how_many_it))}
    mse_testdata_dictionary = {'none':np.zeros((how_many_extras, how_many_it))}
    
    
    for i in range(len(alpha)):
        abs_diff_ss_dictionary[str(alpha[i])] =  np.zeros((how_many_extras, how_many_it))
        mse_testdata_dictionary[str(alpha[i])]  =  np.zeros((how_many_extras, how_many_it))
        
        

    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)

    print('For '+str(type_regression)+ ' regression: \n')
    for iteration in range(how_many_it):
        
     
        
        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
        
    
        y_predicted_unbinned = predict_regressions(X, y, X_test, 'simple')
   
        
        mse_unbinned = mse(y_predicted_unbinned, y_test)
        ss_unbinned = calc_ss(X, y)
        diff = ss_unbinned - ss_unbinned
        abs_diff_ss_dictionary['none'] [0, iteration] = diff**2
        mse_testdata_dictionary['none'] [0, iteration] =  mse_unbinned
        
        for a in alpha:
      
            y_predicted_unbinned = predict_regressions(X, y, X_test, type_regression, a)
       
            mse_unbinned = mse(y_predicted_unbinned, y_test)
            ss_unbinned = calc_ss(X, y)
            diff = ss_unbinned - ss_unbinned
            abs_diff_ss_dictionary[str(a)] [0, iteration] = diff**2
            mse_testdata_dictionary[str(a)] [0, iteration] =  mse_unbinned

        if type_transf in ['binned_centre', 'binned_random', 'rank']:
            list_of_bins = transf(type_transf, list_wanted = extra)
            
        for i in range(how_many_extras - 1):
            
            if type_transf in ['binned_centre', 'binned_random','rank']:
                bins = list_of_bins[i] 
                new_X = data_transf(X, type_transf, bins)
                
            elif type_transf == 'multiplied_non_random':
                new_X = data_transf(X, type_transf, constant = extra[i])
            
            y_predicted_binned = predict_regressions(new_X, y, X_test, 'simple')
            
            mse_binned = mse(y_predicted_binned, y_test)
            ss_binned = calc_ss(new_X, y)
            diff = ss_unbinned - ss_binned
            abs_diff_ss_dictionary['none'] [i+1, iteration] = diff**2
            mse_testdata_dictionary['none'] [i+1, iteration] =  mse_binned
            
            for a in alpha:
                y_predicted_binned = predict_regressions(new_X, y, X_test, type_regression, a)
               
                mse_binned = mse(y_predicted_binned, y_test)
                ss_binned = calc_ss(new_X, y)
                diff = ss_unbinned - ss_binned
                abs_diff_ss_dictionary[str(a)] [i+1, iteration] = diff**2
                mse_testdata_dictionary[str(a)] [i+1, iteration] =  mse_binned
           
    return abs_diff_ss_dictionary, mse_testdata_dictionary

def plotting_against_mse(abs_diff_ss, mse_testdata, type_transf, parameters, size_test, size_train, how_many_it):
    plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
    plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
    plt.ylabel('predictive MSE')
    # if type_transf == 'multiplied_non_random':
    #     log_X = np.mean(abs_diff_ss,axis = 1)
    #     log_y = np.mean(mse_testdata,axis = 1)
    #     a, b = optimise_root(log_X, log_y)
    #     random_x = np.linspace(min(log_X), max(log_X), 100)
    #     plt.plot(random_x, root_function(random_x, a, b))
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
              +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
              +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
    plt.show()
    

def plot_binsize_against_ss(extra, abs_diff_ss_dictionary, type_transf, parameters, size_train, size_test, alpha = None):
    plt.figure()
    plt.plot([0] + list(extra),np.mean(abs_diff_ss_dictionary['none'], axis = 1),'.', label = 'no penalty')
    for a in alpha:
        plt.plot([0] + list(extra),np.mean(abs_diff_ss_dictionary[str(a)], axis = 1),'.', label = '$\\alpha=$' + str(a))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('$E[(S(X) - S(X^{*}))^2]$')
    if type_transf in ['binned_centre', 'binned_random']:
        plt.xlabel('bin size, $h$')
    elif type_transf == 'multiplied_non_random':
        plt.xlabel('$\\epsilon$')
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
                +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
                +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test) )
      
    plt.show()


def plot_ss_against_mse(abs_diff_ss_dictionary, mse_testdata_dictionary, type_transf, parameters, size_train, size_test, alpha = None):


    plt.figure()
    plt.plot(np.mean(abs_diff_ss_dictionary['none'], axis = 1),np.mean(mse_testdata_dictionary['none'] , axis = 1),'.', label = 'no penalty')
    for a in alpha:
        plt.plot(np.mean(abs_diff_ss_dictionary[str(a)], axis = 1),np.mean(mse_testdata_dictionary[str(a)] , axis = 1),'.', label = '$\\alpha=$' + str(a))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
    plt.ylabel('predictive MSE')
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
                +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
                +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test)+'\n Transformation:'+str(type_transf))
    plt.show()
   
    
    
how_many_it = 20
size_test, size_train = 1000, 100
# type_transf = 'multiplied_non_random'
type_transf = 'binned_centre'


if type_transf in ['binned_centre', 'binned_random','rank']:
    max_bin_size = 15
    extra = np.linspace(0.01, max_bin_size, 50)
elif type_transf == 'multiplied_non_random':
    extra = np.linspace(0.01, 5, 50)




# ##### SIMPLE REGRESSION

# type_regression = 'simple'
# abs_diff_ss_dictionary, mse_testdata_dictionary,e1,e2 = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra, type_regression, alpha = [], want_the_betas=True)
# plt.figure()
# plt.plot([0] + list(extra),np.mean(abs_diff_ss_dictionary['none'], axis = 1),'.', label = 'no penalty')
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.ylabel('$E[(S(X) - S(X^{*}))^2]$')
# if type_transf in ['binned_centre', 'binned_random']:
#     plt.xlabel('bin size, $h$')
# elif type_transf == 'multiplied_non_random':
#     plt.xlabel('$\\epsilon$')
# plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
#             +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
#             +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test) )
  
# plt.show()

# plt.figure()
# plt.plot(np.mean(abs_diff_ss_dictionary['none'], axis = 1),np.mean(mse_testdata_dictionary['none'] , axis = 1),'.', label = 'no penalty')
# plt.legend()
# plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
# plt.ylabel('predictive MSE')
# plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta_0$ =' +str(parameters[0])
#             +', $\\beta_1$ = ' +str(parameters[1]) +', $\\mu$ = ' +str(parameters[2]) +' $\\sigma^2$ = ' +str(parameters[3]**2) 
#             +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test)+'\n Transformation:'+str(type_regression))
# plt.show()

# plt.plot([0]+ list(extra), np.mean(e1['none'],axis = 1),'.')



####### RIDGE OR LASSO REGRESSION
# set_of_parameters = [[0.2, 1, 0, 1], [0.2, 0.2, 0, 1], [1, 5, 0, 1], [0.2, 10, 0, 1], [10,10,1,1], [1, 10, 1, 1], [0.2, 10, 1, 1], [1, 100, 1, 1]]
set_of_parameters = [[0.4, 1, 0, 1]]


# alpha = np.array([0.9, 1, 3])
# alpha = np.concatenate((np.array([0.01, 0.1, 0.25, 0.5, 0.75]), alpha))
# alpha = np.array([1])

alpha = np.linspace(0.0001,10,10)
type_regression = 'ridge';




for i in range(len(set_of_parameters)):
    print('Set of parameters: '+str(i)+' out of ' +str(len(set_of_parameters)) )
    parameters  = set_of_parameters[i]
    abs_diff_ss_dictionary, mse_testdata_dictionary = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra, type_regression, alpha)
    

    plot_binsize_against_ss(extra, abs_diff_ss_dictionary, type_transf, parameters, size_train, size_test, alpha)
    plot_ss_against_mse(abs_diff_ss_dictionary, mse_testdata_dictionary, type_transf, parameters, size_train, size_test, alpha)
    
      








