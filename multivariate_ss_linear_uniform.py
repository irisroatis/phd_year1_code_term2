#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 15:49:14 2023

@author: ir318
"""


##### THIS CODE ALLOWS FOR BOTH UNIVARIATE & MULTIVARIATE CASE


import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.tree import DecisionTreeRegressor

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

def standardise(X):
    return (X - np.mean(X)) / np.std(X)

def calc_ss(x,y):
    return x.T @ y

def generate_test_multivariate(e, std, size, beta):
    '''
    Making e, std vectors such that X1 (first column of X) is generated
    using N(e1,std1) and so on.
    
    beta is the vector of all parameters (including intercept)

    '''
    what_dimension = len(e)
    X = np.zeros((size, what_dimension+1))
    X[:,0] = 1
    for i in range(what_dimension):
        X[:,i+1] = np.random.uniform(e[i], std[i], size)
    y = X @ beta + np.random.normal(0, 1, size)
    X = X[:,1:]
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
    
def multivariate_ss_against_mse(how_many_it, parameter_dictionary, size_test, size_train, type_transf, extra=None):
    beta = parameter_dictionary['beta']
    lower =  parameter_dictionary['lower']
    upper =  parameter_dictionary['upper']

    how_many_extras = len(extra) + 1


    difference_ss = np.zeros((how_many_extras, how_many_it))
    abs_diff_ss =  np.zeros((how_many_extras, how_many_it))
    mse_testdata =  np.zeros((how_many_extras, how_many_it))
     
    # generating test data the same for all binnings and iterations
    X_test, y_test = generate_test_multivariate(lower, upper, size_test, beta)

    
    for iteration in range(how_many_it):
        
        print((iteration+1)/how_many_it)    # code progress
        
        X,y = generate_test_multivariate(lower, upper, size_train, beta)
    
        regressor = LinearRegression()  
  
        regressor.fit(X, y) #training the algorithm
        y_predicted_unbinned = regressor.predict(X_test)
            
        mse_unbinned = mse(y_predicted_unbinned, y_test)
        
        ss_unbinned = np.zeros((X.shape[1],1))
        for index in range(X.shape[1]):
            ss_unbinned[index] =   calc_ss(X[:,index], y)
        
        diff = np.linalg.norm(ss_unbinned-ss_unbinned)
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
            
            regressor.fit(new_X, y) #training the algorithm
            y_predicted_binned = regressor.predict(X_test)
                
         
            mse_binned = mse(y_predicted_binned, y_test)
        
            ss_binned = np.zeros((new_X.shape[1],1))
            for index in range(X.shape[1]):
                ss_binned[index] =   calc_ss(new_X[:,index], y)
            
            diff = np.linalg.norm(ss_unbinned-ss_binned)
            difference_ss[i+1,iteration],  abs_diff_ss[i+1,iteration] = diff, diff**2
       
            mse_testdata[i+1,iteration] = mse_binned
    
    return difference_ss, standardise(abs_diff_ss), mse_testdata

def plotting_against_mse(abs_diff_ss, mse_testdata, type_transf, parameter_dictionary, size_test, size_train, how_many_it):
    beta = parameter_dictionary['beta']
    e =  parameter_dictionary['mean']
    std =  parameter_dictionary['std_dev']
    
    plt.figure()
    plt.scatter(np.mean(abs_diff_ss,axis = 1), np.mean(mse_testdata,axis = 1))
    plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
    plt.ylabel('predictive MSE')
    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta$ =' +str(beta)
             +', $\\mu$ = ' +str(e) +' $\\sigma$ = ' +str(std) 
              +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
    plt.show()
    
def plotting_width_against_ss(abs_diff_ss, extra, type_transf, parameter_dictionary, size_test, size_train, how_many_it):
    beta = parameter_dictionary['beta']
    e =  parameter_dictionary['mean']
    std =  parameter_dictionary['std_dev']
    
    plt.figure()
    plt.scatter([0] + list(extra), np.mean(abs_diff_ss,axis = 1))
    plt.ylabel('$E[(S(X) - S(X^{*}))^2]$')
    if type_transf in ['binned_centre', 'binned_random']:
        plt.xlabel('bin size, $h$')
    elif type_transf == 'multiplied_non_random':
        plt.xlabel('$\\epsilon$')

    plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta$ =' +str(beta)
             + ', $\\mu$ = ' +str(e) +' $\\sigma$ = ' +str(std) 
              +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
    


how_many_it = 300
size_test, size_train = 1000, 100
parameter_dictionary = {};
# parameter_dictionary["beta"] = np.random.rand(5);
# parameter_dictionary["mean"] = random.sample(range(0, 10), k=4);
# parameter_dictionary["std_dev"] = random.choices(range(1, 5), k=4);
# parameter_dictionary["beta"] = [0.4, 1, 0.2, 0.5, 0.8, 0.9, 0.2, 0.1, 2, 0.4, 0.55];
# parameter_dictionary["mean"] = [0, 1, 5, 2, 4, 2, 7, 3, 0, 6];
# parameter_dictionary["std_dev"] = [1, 3, 1, 2, 3, 5, 1, 2, 4, 2];


trial_1 = [0];
trial_2 = [3];
trial_3 = [5];



parameter_dictionary["beta"] = [1.5, 2];
parameter_dictionary["lower"] = trial_1;
parameter_dictionary["upper"] = [6];

# type_transf = 'multiplied_non_random'
type_transf = 'binned_centre'


if type_transf in ['binned_centre', 'binned_random']:
    max_bin_size = 15
    extra = np.linspace(0.01, max_bin_size, 70)
elif type_transf == 'multiplied_non_random':
    extra = np.linspace(0.01, 5, 50)




difference_ss, abs_diff_ss, mse_testdata = multivariate_ss_against_mse(how_many_it, parameter_dictionary, size_test, size_train, type_transf, extra)

binsize_list = [0] + list(extra)
length_binsize_list = len(binsize_list)
ss_list = np.mean(abs_diff_ss,axis = 1)
mse_list = np.mean(mse_testdata,axis = 1)

up_until = length_binsize_list //3


############## WHEN SINGLE PLOTS ARE WANTED
# plotting_against_mse(abs_diff_ss, mse_testdata, type_transf, parameter_dictionary, size_test, size_train, how_many_it)
# plotting_width_against_ss(abs_diff_ss, extra, type_transf, parameter_dictionary, size_test, size_train, how_many_it)


############## IF ZOOMED IN PLOTS ARE WANTED
# plt.figure()
# plt.subplot(221)
# plt.plot(binsize_list,ss_list,'.')

# plt.subplot(222)
# plt.plot(binsize_list[:up_until],ss_list[:up_until],'.')

# plt.subplot(223)
# plt.plot(binsize_list[up_until:up_until*2],ss_list[up_until:up_until*2],'.')

# plt.subplot(224)
# plt.plot(binsize_list[up_until*2:],ss_list[up_until*2:],'.')

# plt.suptitle('bin size $h$ against $E[(S(X) - S(X^{*}))^2]$, dimension: '+str(len(parameter_dictionary["std_dev"])))
# plt.show()


# plt.figure()
# plt.subplot(221)
# plt.plot(ss_list,mse_list,'.')

# plt.subplot(222)
# plt.plot(ss_list[:up_until],mse_list[:up_until],'.')

# plt.subplot(223)
# plt.plot(ss_list[up_until:up_until*2],mse_list[up_until:up_until*2],'.')

# plt.subplot(224)
# plt.plot(ss_list[up_until*2:],mse_list[up_until*2:],'.')

# plt.suptitle('$E[(S(X) - S(X^{*}))^2]$ against predictive MSE, dimension: '+str(len(parameter_dictionary["std_dev"])))
# plt.show()

############# COMPARING VARIOUS PARAMETER VALUES 

###### plot bin size against diff squared

plt.figure()
plt.plot(ss_list,mse_list,'.',label ='lower ='  + str(parameter_dictionary["lower"]))


parameter_dictionary["lower"] = trial_2
difference_ss2, abs_diff_ss2, mse_testdata2 = multivariate_ss_against_mse(how_many_it, parameter_dictionary, size_test, size_train, type_transf, extra)
ss_list2 = np.mean(abs_diff_ss2,axis = 1)
mse_list2 = np.mean(mse_testdata2,axis = 1)
plt.plot(ss_list2,mse_list2,'.',label ='lower ='  + str(parameter_dictionary["lower"]))


parameter_dictionary["lower"] = trial_3
difference_ss3, abs_diff_ss3, mse_testdata3 = multivariate_ss_against_mse(how_many_it, parameter_dictionary, size_test, size_train, type_transf, extra)
ss_list3 = np.mean(abs_diff_ss3,axis = 1)
mse_list3 = np.mean(mse_testdata3,axis = 1)
plt.plot(ss_list3,mse_list3,'.',label ='lower =' + str(parameter_dictionary["lower"]))


plt.xlabel('$E[(S(X) - S(X^{*}))^2]$')
plt.ylabel('predictive MSE')
plt.legend()
plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta$ = ' +str(parameter_dictionary["beta"] ) +' upper= ' +str(parameter_dictionary["upper"] ) 
          +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
plt.show()

############# COMPARING VARIOUS PARAMETER VALUES

###### plot diff squared against MSE

plt.figure()
parameter_dictionary["lower"] = trial_1
plt.plot(binsize_list,ss_list,'.',label ='lower ='  + str(parameter_dictionary["lower"]))


parameter_dictionary["lower"] = trial_2
plt.plot(binsize_list,ss_list2,'.',label ='lower ='  + str(parameter_dictionary["lower"]))


parameter_dictionary["lower"] = trial_3
plt.plot(binsize_list,ss_list3,'.',label ='lower =' + str(parameter_dictionary["lower"]))

plt.xlabel('bin size, $h$')
plt.ylabel('$E[(S(X) - S(X^{*}))^2]$')
plt.legend()
plt.title('Linear Regression - Transformation: '+type_transf+', \n Parameters: $\\beta$ = ' +str(parameter_dictionary["beta"] ) +' upper = ' +str(parameter_dictionary["upper"] ) 
          +'\n Sizes: train: ' +str(size_train)+', test: ' +str(size_test))
plt.show()