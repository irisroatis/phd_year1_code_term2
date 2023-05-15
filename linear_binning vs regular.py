#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:27:38 2023

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


def generate_test(e, std, size, beta_0, beta_1):
    X = np.random.normal(e, std, size)
    y = beta_0 + beta_1 * X 
    return X, y

def transf(type_transf, list_wanted = None):
    if type_transf in ['binned_centre', 'binned_random','rank']:
        list_of_bins = []
        for i in range(len(list_wanted)):
            bin_size = list_wanted[i]
            bins = np.arange(0,1000,bin_size)
            bins = np.concatenate((-bins[::-1][:-1], bins))
            list_of_bins.append(bins)
        return list_of_bins
    
def data_transf(X, type_transf, bins = None, constant = None):
    if type_transf in ['binned_centre', 'binned_random','rank']:
        return put_in_bins(X, bins, type_transf)
    elif type_transf == 'multiplied_non_random':
        return (1 + constant) * X
    
    
def predict_regressions(X, y, X_test, type_regression, a = None):
    if type_regression =='ridge':
        regressor = Ridge(alpha = a,fit_intercept = True)  
    elif type_regression == 'lasso':
        regressor = Lasso(alpha = a)
    elif type_regression == 'simple':
        regressor = LinearRegression()      
    elif type_regression == 'cart':
        regressor = DecisionTreeRegressor()  
        
    regressor.fit(X.reshape(-1,1), y) #training the algorithm
    
    y_predicted = regressor.predict(X_test.reshape(-1,1))
    
    return y_predicted, regressor.intercept_, regressor.coef_
    

def st (X):
    mean = np.mean(X)
    std = np.std(X)
    norm_X =  (X - mean) / std
    return norm_X
    
def ss_against_mse(how_many_it, parameters, size_test, size_train, type_transf,standardise, extra=None, type_regression = None, alpha = None):
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]
    
    how_many_extras = len(extra) + 1
    

    estimated_betas_0 = {'none':np.zeros((how_many_extras, how_many_it))}
    estimated_betas_1 = {'none':np.zeros((how_many_extras, how_many_it))}


    for i in range(len(alpha)):
       
        estimated_betas_0[str(alpha[i])] =  np.zeros((how_many_extras, how_many_it))
        estimated_betas_1[str(alpha[i])] =  np.zeros((how_many_extras, how_many_it))


    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)
    print(X_test)
    print(y_test)

    print('For '+str(type_regression)+ ' regression: \n')
    for iteration in range(how_many_it):

        
        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
        print(X)
        print(y)
        if standardise:
            norm_X = st(X)
        else:
            norm_X = X
   
        y_predicted_unbinned, estimated_betas_0['none'] [0, iteration],  estimated_betas_1['none'] [0, iteration] = predict_regressions(norm_X, y, X_test, 'simple')
        

        for a in alpha:
            y_predicted_unbinned,  estimated_betas_0[str(a)] [0, iteration],  estimated_betas_1[str(a)] [0, iteration] = predict_regressions(norm_X, y, X_test, type_regression, a)

       
        if type_transf in ['binned_centre', 'binned_random', 'rank']:
            list_of_bins = transf(type_transf, list_wanted = extra)
            
        for i in range(how_many_extras - 1):
            
           bins = list_of_bins[i] 
           new_X = data_transf(X, type_transf, bins)
           
           if standardise:
               norm_X = st(new_X)
           else:
               norm_X = new_X
                
           y_predicted_binned, estimated_betas_0['none'] [i+1, iteration],  estimated_betas_1['none'] [i+1, iteration] = predict_regressions(norm_X, y, X_test, 'simple')
                    
           # for a in alpha:
           #     y_predicted_binned, estimated_betas_0[str(a)] [i+1, iteration], estimated_betas_1[str(a)] [i+1, iteration]  = predict_regressions(norm_X, y, X_test, type_regression, a)
               
    return estimated_betas_0, estimated_betas_1


def plotting_betas(extra, type_regression, alpha, e1, e2):
    if type_regression in ['lasso', 'ridge']:

            plt.figure()
            for a in alpha:
                plt.plot([0]+ list(extra), np.mean(e1[str(a)],axis = 1),'.', label = '$\\alpha =$'+str(a))
            plt.plot([0]+ list(extra), np.mean(e1['none'],axis = 1),'.', label = 'no penalty')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('true $\\beta_0$ =' +str(parameters[0])+' and X $\sim$ N'+str(parameters[2:]))
            plt.xlabel('bin_size')
            plt.ylabel('$\\beta_0$')
            plt.show()
                
            
            plt.figure()
            for a in alpha:
                plt.plot([0]+ list(extra), np.mean(e2[str(a)],axis = 1), '.', label = '$\\alpha =$'+str(a))
            plt.plot([0]+ list(extra), np.mean(e2['none'],axis = 1),'.', label = 'no penalty')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.title('true $\\beta_1$ =' +str(parameters[1])+' and X $\sim$ N'+str(parameters[2:]))
            plt.xlabel('bin_size')
            plt.ylabel('$\\beta_1$')
            plt.show()
            
            # plt.figure()
            # for a in alpha:
            #     plt.plot([0]+ list(extra), np.var(e1[str(a)],axis = 1),'.', label = '$\\alpha =$'+str(a))
            # plt.plot([0]+ list(extra), np.var(e1['none'],axis = 1),'.', label = 'no penalty')
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.title('true $\\beta_0$ =' +str(parameters[0])+' and X $\sim$ N'+str(parameters[2:]))
            # plt.xlabel('bin_size')
            # plt.ylabel('$var(\\beta_0$)')
            # plt.show()
                
            
            # plt.figure()
            # for a in alpha:
            #     plt.plot([0]+ list(extra), np.var(e2[str(a)],axis = 1), '.', label = '$\\alpha =$'+str(a))
            # plt.plot([0]+ list(extra), np.var(e2['none'],axis = 1),'.', label = 'no penalty')
            # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # plt.title('true $\\beta_1$ =' +str(parameters[1])+' and X $\sim$ N'+str(parameters[2:]))
            # plt.xlabel('bin_size')
            # plt.ylabel('$var(\\beta_1$)')
            # plt.show()
    
    else:
        print('Not lasso or ridge')
        


    
how_many_it = 100
size_test, size_train = 100, 50
# type_transf = 'multiplied_non_random'
type_transf = 'binned_centre'


if type_transf in ['binned_centre', 'binned_random','rank']:
    max_bin_size = 10
    
    extra = np.linspace(0, max_bin_size, 100)
elif type_transf == 'multiplied_non_random':
    extra = np.linspace(0, 5, 50)







####### RIDGE REGRESSION
# set_of_parameters = [[0.2, 1, 0, 1], [0.2, 0.2, 0, 1], [1, 5, 0, 1], [0.2, 10, 0, 1], [10,10,1,1], [1, 10, 1, 1], [0.2, 10, 1, 1], [1, 100, 1, 1]]
set_of_parameters = [[0.4, 1, 0, 1]]


# alpha = np.array([0.9, 1, 3])
# alpha = np.concatenate((np.array([0.01, 0.1, 0.25, 0.5, 0.75]), alpha))
# alpha = np.array([1])

alpha = np.linspace(0,100,50)
type_regression = 'ridge';
standardise = True


for p in range(len(set_of_parameters)):
    print('Set of parameters: '+str(p)+' out of ' +str(len(set_of_parameters)) )
    parameters  = set_of_parameters[p]
    
    beta_0_true = parameters[0]
    beta_1_true = parameters[1]
    e = parameters[2]
    std = parameters[3]

    estimated_beta0 = np.zeros((len(extra),len(alpha)))
    estimated_beta1 =  np.zeros((len(extra),len(alpha)))

    
    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test(e, std, size_test, beta_0_true, beta_1_true)

    for iteration in range(how_many_it):

        X,y = generate_test(e, std, size_train, beta_0_true, beta_1_true)
 
        if standardise:
            norm_X = st(X)
        else:
            norm_X = X
        
        trial_X = np.hstack((np.ones((size_train,1)),norm_X.reshape(-1,1)))
        same_1 = trial_X.T @ trial_X
        same_2 = trial_X.T @ y
        beta_0, beta_1 = np.linalg.inv(same_1) @ same_2
        
        estimated_beta0[0,0] += beta_0
        estimated_beta1[0,0] += beta_1
        
        for index_alpha in range(1,len(alpha)):
            a = alpha[index_alpha]
            beta_0, beta_1  = np.linalg.inv(same_1 + a * np.identity(2)) @ same_2
            estimated_beta0[0, index_alpha] += beta_0
            estimated_beta1[0, index_alpha] += beta_1
            
            
        if type_transf in ['binned_centre', 'binned_random', 'rank']:
            list_of_bins = transf(type_transf, list_wanted = extra[1:])
        
        
        for index_bin in range(1, len(extra)):
            bins = list_of_bins[index_bin-1] 
            new_X = data_transf(X, type_transf, bins)
            
            if standardise:
                norm_X = st(new_X)
            else:
                norm_X = new_X

            trial_X_2 = np.hstack((np.ones((size_train,1)),norm_X.reshape(-1,1)))
            
            beta_0, beta_1 = np.linalg.inv(trial_X_2.T @ trial_X_2) @ trial_X_2.T @ y
            estimated_beta0[index_bin, 0] += beta_0
            estimated_beta1[index_bin, 0] += beta_1
           # for a in alpha:
           #     y_predicted_binned, estimated_betas_0[str(a)] [i+1, iteration], estimated_betas_1[str(a)] [i+1, iteration]  = predict_regressions(norm_X, y, X_test, type_regression, a)
               
    estimated_beta0 /= how_many_it
    estimated_beta1 /= how_many_it
    
    plt.plot(estimated_beta0[0,1:],estimated_beta1[0,1:], 'r.',label='varying $\\alpha$')
    plt.plot(estimated_beta0[1:,0],estimated_beta1[1:,0], 'b.',label='varying $h$')
    plt.plot(estimated_beta0[0,0],estimated_beta1[0,0], 'gx', label = '$\\alpha = 0, h = 0$')
    plt.legend()
    plt.xlabel('$\\beta_0$')
    plt.ylabel('$\\beta_1$')
    plt.title('True $\\beta_0 = $'+str(parameters[0])+', true $\\beta_1=$'+str(parameters[1])+', X~N'+str(parameters[2:])+'\n standardised: '+str(standardise))
    plt.show()
    
    # plotting_betas(extra, type_regression, alpha, e1, e2)
    
    # beta0_nopenalty = np.mean(e1['none'],axis = 1)
    # beta1_nopenalty = np.mean(e2['none'],axis = 1)
    
    # beta0_nopen_nobin = beta0_nopenalty[0]
    # beta1_nopen_nobin = beta1_nopenalty[1]
    
    # beta0_nobinning = np.zeros_like(alpha)
    # beta1_nobinning = np.zeros_like(alpha)
    
    # for index in range(len(alpha)):
    #     a = alpha[index]
    #     beta0_nobinning[index] = np.mean(e1[str(a)][0,:])
    #     beta1_nobinning[index] = np.mean(e2[str(a)][0,:])
        
    
    # plt.plot(beta0_nobinning[1:], beta1_nobinning[1:], 'r.',label='varying $\\alpha$')  
    # plt.plot(beta0_nopenalty, beta1_nopenalty, 'b.', label = 'varying $h$')  
    # plt.plot(beta0_nopen_nobin, beta1_nopen_nobin, 'gx', label = '$\\alpha = 0, h = 0$')
    # plt.legend()
    # plt.xlabel('$\\beta_0$')
    # plt.ylabel('$\\beta_1$')
    # plt.title('True $\\beta_0 = $'+str(parameters[0])+', true $\\beta_1=$'+str(parameters[1])+', X~N'+str(parameters[2:])+'\n standardised: '+str(standardise))
    # plt.show()




