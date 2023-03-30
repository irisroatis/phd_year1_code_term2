#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 13:03:01 2023

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
    p0 = [max(y), 0, 0, 0]
    logistic_params, _ = curve_fit(logistic_curve, X, y, p0 = p0)
    return logistic_params


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
    # X = np.random.normal(e, std, size)
    X = np.random.uniform(e, std, size)
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
    
    
def predict_regressions(X, y, X_test, type_regression, a = None, want_the_beta = False):
    if type_regression =='ridge':
        regressor = Ridge(alpha = a)  
    elif type_regression == 'lasso':
        regressor = Lasso(alpha = a)
    elif type_regression == 'simple':
        regressor = LinearRegression()      
    elif type_regression == 'cart':
        regressor = DecisionTreeRegressor()  
        
        
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    
    # plt.figure(figsize=(25,20))
    # _ = tree.plot_tree(regressor,filled=True)
    # plt.show()
    
    y_predicted = regressor.predict(X_test.reshape(-1,1))
    if not want_the_beta:
        return y_predicted
    else:
        return y_predicted, regressor.intercept_, regressor.coef_
    
def ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra=None, type_regression = None, alpha = None, want_the_betas = False):
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]
    
    how_many_extras = len(extra) + 1
    
    if want_the_betas:
        estimated_betas_0 = {'none':np.zeros((how_many_extras, how_many_it))}
        estimated_betas_1 = {'none':np.zeros((how_many_extras, how_many_it))}


    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)
    
    sample_means = []
    
    for iteration in range(how_many_it):
            
        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
        
        sample_means.append(np.mean(X))

        if want_the_betas:
            y_predicted_unbinned, estimated_betas_0['none'] [0, iteration],  estimated_betas_1['none'] [0, iteration] = predict_regressions(X, y, X_test, 'simple',want_the_beta =  True)
        else:
            y_predicted_unbinned = predict_regressions(X, y, X_test, 'simple',want_the_beta =  False)

    

     
        if type_transf in ['binned_centre', 'binned_random', 'rank']:
            list_of_bins = transf(type_transf, list_wanted = extra)
            
        for i in range(how_many_extras - 1):
            
            if type_transf in ['binned_centre', 'binned_random','rank']:
                bins = list_of_bins[i] 
                new_X = data_transf(X, type_transf, bins)
        
      
            if want_the_betas:
                y_predicted_binned, estimated_betas_0['none'] [i+1, iteration], estimated_betas_1['none'] [i+1, iteration] = predict_regressions(new_X, y, X_test, 'simple',want_the_beta =  True)
            else:
                y_predicted_binned = predict_regressions(new_X, y, X_test, 'simple',want_the_beta =  False)

    return estimated_betas_0, estimated_betas_1, sample_means



how_many_it = 300
size_test, size_train = 1000, 100
type_transf = 'binned_centre'


max_bin_size = 15
extra = np.linspace(0.01, max_bin_size, 70)

# table = [['Iter_nr','True beta0', 'True beta1', 'Mean', 'Inf Point x', 'Supremum','My Guess Supr', 'Convergence Value']]

# table2 = [['Iter_nr','True beta0', 'True beta1', 'Mean','Convergence Value']]

table = [['Iter_nr','True beta0', 'True beta1', 'Low','High', 'Inf Point x', 'Supremum','My Guess Supr', 'Convergence Value']]

table2 = [['Iter_nr','True beta0', 'True beta1','Low','High','Convergence Value']]



for i in range(30):
    true_b0 = np.round(random.uniform(0, 2),1);
    true_b1 = np.round(random.uniform(0, 2),1);
    # true_mean = np.round(random.uniform(0, 2),1);
    true_low = np.round(random.uniform(-1, 1),1);
    true_up = np.round(random.uniform(2, 4),1);
    
    # parameters = [true_b0, true_b1, true_mean, 1];
    parameters = [true_b0, true_b1, true_low, true_up];
    
    print(parameters)
    
    type_regression = 'simple'
    e1, e2, sample_means = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra, type_regression, alpha = [], want_the_betas=True)
    value_intercepts = np.mean(e1['none'],axis = 1)
    value_gradients = np.mean(e2['none'],axis = 1)
    bin_sizes = [0]+ list(extra)
    
    try:
        param_log = optimise_logistic(bin_sizes, value_intercepts)
        error = False
    except RuntimeError:
        error = True;
        
    # if error:
    #     table.append([i, true_b0, true_b1, true_mean, 'Error', 'Error', true_b0 + true_b1 * np.mean(sample_means) , value_intercepts[-1]])
    # else:
    #     table.append([i, true_b0, true_b1, true_mean, param_log[2],  param_log[0] +  param_log[1], true_b0 + true_b1 *  np.mean(sample_means), value_intercepts[-1]])
    
    if error:
        table.append([i, true_b0, true_b1, true_low, true_up, 'Error', 'Error', true_b0 + true_b1 * np.mean(sample_means) , value_intercepts[-1]])
    else:
        table.append([i, true_b0, true_b1, true_low, true_up, param_log[2],  param_log[0] +  param_log[1], true_b0 + true_b1 * np.mean(sample_means), value_intercepts[-1]])
    
    
        plt.figure()
        plt.plot(bin_sizes, value_intercepts,'.')
        plt.plot(bin_sizes, logistic_curve(bin_sizes, param_log[0], param_log[1], param_log[2], param_log[3]))
        plt.title('Parameters: ' +str(parameters))
        # plt.savefig('Experiment_plots/plot'+str(i)+'.png')
        plt.show()
        
    plt.figure()
    plt.plot(bin_sizes, value_gradients,'.')
    plt.title('Parameters: ' +str(parameters))
    
    # plt.savefig('Experiment_plots/plot_beta1_'+str(i)+'.png')
    plt.show()
        
    table2.append([i, true_b0, true_b1, true_low, true_up, value_gradients[-1]])
        
from tabulate import tabulate
print(tabulate(table))

print(tabulate(table2))


# true_b0 = 8;
# true_b1 = 2;
# true_mean = 0.1;
# parameters = [true_b0, true_b1, true_mean, 1];

# type_regression = 'simple'
# e1, e2 = ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf, extra, type_regression, alpha = [], want_the_betas=True)
# value_intercepts = np.mean(e1['none'],axis = 1)
# bin_sizes = [0]+ list(extra)
# param_log = optimise_logistic(bin_sizes, value_intercepts)

# plt.plot(bin_sizes, value_intercepts,'.')
# plt.plot(bin_sizes, logistic_curve(bin_sizes, param_log[0], param_log[1], param_log[2], param_log[3]))
# plt.title('True param: '+str(parameters))
# plt.show()