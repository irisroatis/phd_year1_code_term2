#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:24:39 2023

@author: roatisiris
"""



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import StandardScaler

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

def plotting_betas(extra, alpha, e1, e2):

        plt.figure()
        for a in alpha:
            plt.plot([0]+ list(extra), np.mean(e1[str(a)],axis = 1),'.', label = '$\\alpha =$'+str(a))
        plt.plot([0]+ list(extra), np.mean(e1['none'],axis = 1),'.', label = 'no penalty')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('bin_size')
        plt.ylabel('$\\beta_0$')
        plt.show()
            
        
        plt.figure()
        for a in alpha:
            plt.plot([0]+ list(extra), np.mean(e2[str(a)],axis = 1), '.', label = '$\\alpha =$'+str(a))
        plt.plot([0]+ list(extra), np.mean(e2['none'],axis = 1),'.', label = 'no penalty')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('bin_size')
        plt.ylabel('$\\beta_1$')
        plt.show()
        
def plot_connection_bin_penalty(extra, alpha, e1, e2):
        
    corresponding_bin_sizes0 = np.zeros_like(alpha)   
    corresponding_bin_sizes1 = np.zeros_like(alpha)   

    beta0_nopenalty = np.mean(e1['none'],axis = 1)
    beta1_nopenalty = np.mean(e2['none'],axis = 1)
    
    extra = np.concatenate((np.array([0]),extra))
    
    for index in range(len(alpha)):
        a = alpha[index]
        
        aim0 = np.mean(e1[str(a)],axis = 1)[0] 
        diff_parameter0 = beta0_nopenalty - aim0
        corresponding_bin_sizes0[index] = extra[np.argmin(abs(diff_parameter0))]
      
        
        aim1 = np.mean(e2[str(a)],axis = 1)[0] 
        diff_parameter1 = beta1_nopenalty - aim1
        corresponding_bin_sizes1[index] = extra[np.argmin(abs(diff_parameter1))]
       
            
    plt.plot(alpha, corresponding_bin_sizes0,'.',label='for $\\beta_0$')
    plt.plot(alpha, corresponding_bin_sizes1,'.',label='for $\\beta_1$')
    plt.xlabel('$\\alpha$')
    plt.ylabel('corresponding $h$')
    plt.legend()
    plt.title('Connection')
    plt.show()


    # print(tabulate(table0))
    
    return corresponding_bin_sizes0, corresponding_bin_sizes1


size_train1 = 500
size_train2 = 500
size_test1 = 100
size_test2 = 100

e1 = 0
e2 = 1
std1 = 1
std2 = 1

standardise = True

alpha = np.linspace(0.0001,100,100)

list_bin_sizes = np.linspace(0.01, 20, 100)

how_many_extras = len(list_bin_sizes) + 1

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
    bins = np.arange(0,1000,bin_size)
    bins = np.concatenate((-bins[::-1][:-1], bins))
    list_of_bins.append(bins)
    

estimated_betas_0 = {'none':np.zeros((how_many_extras, how_many))}
estimated_betas_1 = {'none':np.zeros((how_many_extras, how_many))}

for a in alpha:
    estimated_betas_0[str(a)] =np.zeros((how_many_extras, how_many))
    estimated_betas_1[str(a)] =np.zeros((how_many_extras, how_many))
  
X_test, y_test = generate_test(e1, std1, e2, std2, size_test1, size_test2)    

for iteration in range(how_many):    
    print((iteration+1)/how_many)
    X,y = generate_test(e1, std1, e2, std2, size_train1, size_train2)
    
    if standardise:
        std_X = StandardScaler().fit_transform(X.reshape(-1,1)) 
    else:
        std_X = X
    
    regressor = LogisticRegression(penalty='none')
    regressor.fit(std_X.reshape(-1,1), y) #training the algorithm
    # y_predicted_unbinned = regressor.predict(X_test.reshape(-1,1))
    estimated_betas_0['none'][0, iteration] = regressor.intercept_
    estimated_betas_1['none'][0, iteration] = regressor.coef_
    
    for a in alpha:

        regressor = LogisticRegression(penalty='l1', C = 1/a, solver="saga")
        regressor.fit(std_X.reshape(-1,1), y) #training the algorithm
        estimated_betas_0[str(a)][0, iteration] = regressor.intercept_
        estimated_betas_1[str(a)][0, iteration] = regressor.coef_
        

    
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(X, bins)
        
        if standardise:
            std_new_X = StandardScaler().fit_transform(new_X.reshape(-1,1))
        else:
            std_new_X = new_X
      
        regressor = LogisticRegression(penalty='none')
        regressor.fit(std_new_X.reshape(-1,1), y) #training the algorithm
        # y_predicted_binned = regressor.predict(X_test.reshape(-1,1))
        estimated_betas_0['none'][i+1, iteration] = regressor.intercept_
        estimated_betas_1['none'][i+1, iteration] = regressor.coef_
        
        # for a in alpha:
        #     regressor = LogisticRegression(penalty='l2', C = 1/a)
        #     regressor.fit(std_new_X.reshape(-1,1), y) #training the algorithm
        #     estimated_betas_0[str(a)][i+1, iteration] = regressor.intercept_
        #     estimated_betas_1[str(a)][i+1, iteration] = regressor.coef_
            



plotting_betas(list_bin_sizes, alpha, estimated_betas_0, estimated_betas_1)

beta0_nopenalty = np.mean(estimated_betas_0['none'],axis = 1)
beta1_nopenalty = np.mean(estimated_betas_1['none'],axis = 1)

beta0_nopen_nobin = beta0_nopenalty[0]
beta1_nopen_nobin = beta1_nopenalty[1]

beta0_nobinning = np.zeros_like(alpha)
beta1_nobinning = np.zeros_like(alpha)

for index in range(len(alpha)):
    a = alpha[index]
    beta0_nobinning[index] = np.mean(estimated_betas_0[str(a)][0,:])
    beta1_nobinning[index] = np.mean(estimated_betas_1[str(a)][0,:])
    

plt.plot(beta0_nobinning[1:], beta1_nobinning[1:], 'r.',label='varying $\\alpha$')  
plt.plot(beta0_nopenalty, beta1_nopenalty, 'b.', label = 'varying $h$')  
plt.plot(beta0_nopen_nobin, beta1_nopen_nobin, 'gx', label = '$\\alpha = 0, h = 0$')
plt.legend()
plt.xlabel('$\\beta_0$')
plt.ylabel('$\\beta_1$')
plt.title('Logistic regression, standardise: '+ str(standardise))
plt.show()
