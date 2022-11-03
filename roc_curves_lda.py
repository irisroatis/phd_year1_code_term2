#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:24:25 2022

@author: roatisiris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
import scipy.stats
from mpl_toolkits import mplot3d
import random
from defined_functions import *


####### ESTABLISHING PARAMETER VALUES

# mu1 = np.array([0, 0]) # mean and standard deviation
# sigma1 = np.array([[1, 0],[ 0,1]]) # mean and standard deviation
# mu2 = np.array([1, 1]) # mean and standard deviation
# sigma2 = np.array([[1, 0],[ 0,1]]) # mean and standard deviation

mu1 = np.array([0]) # mean and standard deviation
sigma1 = np.array([1]) # mean and standard deviation
mu2 = np.array([1]) # mean and standard deviation
sigma2 = np.array([1]) # mean and standard deviation

dim = len(mu1) # dimension of the problem

iterations = 1000 # length of each sample in a single experiment
how_many_times_repeat = 1 # how many experiments wanted

value_b = 1 # approximate bin size
value_nt = 20 # size of training sample from one normal




###### SIMULATING DATA FOR TESTING

testing_data, belonging_classes = generating_test_data(how_many_times_repeat, 
                                                       iterations, mu1, sigma1, 
                                                       mu2, sigma2, 
                                                       plot_classes = True)




###### TRAINING THE MODELS


    
size1 = value_nt
size2 = value_nt
bin_size = value_b

# generate from normal distrib
if dim == 1:
    s1 = np.random.normal(mu1, sigma1, size1)
    s2 = np.random.normal(mu2, sigma2, size2)
    
else:
    s1 = np.random.multivariate_normal(mu1, sigma1, size1)
    s2 = np.random.multivariate_normal(mu2, sigma2, size2)

# bins are created around 0 (one bin is set with ends in 0 and outwards
# from there)
# bins = np.arange(0,100,bin_size)
# bins = np.concatenate((-bins[::-2], bins))

bins = np.linspace(-100,100,int(200//bin_size))


s1_binned = put_in_bins(s1, bins)
s2_binned = put_in_bins(s2, bins)



## this is our own implementation of 'trainig the model' using LDA

alpha, c = lda_find_hyperplane(s1_binned,s2_binned)
print('Decision boundary ' + str(c))


plt.figure(figsize=(18, 6), dpi=80)
plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.hist(s1, alpha=0.5, label='N('+str(mu1[0])+','+str(sigma1[0])+')')
plt.hist(s2, alpha=0.5, label='N('+str(mu2[0])+','+str(sigma2[0])+')')
plt.legend(loc='upper right')
plt.title('training data before binning')


plt.subplot(1, 2, 2) # row 1, col 2 index 1
plt.hist(s1_binned, alpha=0.5, label='N('+str(mu1[0])+','+str(sigma1[0])+')')
plt.hist(s2_binned, alpha=0.5, label='N('+str(mu2[0])+','+str(sigma2[0])+')')
plt.axvline(x = c/alpha, color = 'r', label = 'boundary')
plt.legend(loc='upper right')
plt.title('training data after binning')
plt.show()


##### TRY MODELS ON TEST DATA

list_tpr, list_fpr, list_accuracy = list(), list(), list()
# list_tpr_b, list_fpr_b = list(), list()

try_values_for_t = np.linspace(-20,20,200)

for t in try_values_for_t:
    
    tpr = np.zeros((how_many_times_repeat,))
    fpr = np.zeros((how_many_times_repeat,))
    accuracy = np.zeros((how_many_times_repeat,))
    
    # tpr_b = np.zeros((how_many_times_repeat,))
    # fpr_b = np.zeros((how_many_times_repeat,))
    
    
    for repeat in range(how_many_times_repeat):
        
        testing_model_on_binned = put_in_bins(testing_data[repeat],bins)
        testing_model_on_unbinned = testing_data[repeat]
        correct_classes = belonging_classes[repeat]


        predicted_classes_unbinnned = np.zeros((iterations,))
        predicted_classes_binnned = np.zeros((iterations,))
        for itera in range(iterations):
            predicted_classes_unbinnned[itera,] = lda_predict(testing_model_on_unbinned[itera,], alpha, t * alpha)
            # predicted_classes_binnned[itera,] = lda_predict(testing_model_on_binned[itera,], alpha, c)

        
        tpr[repeat,], fpr[repeat,], accuracy[repeat,]= tprfpr(correct_classes, predicted_classes_unbinnned, True)
        # tpr_b[repeat,], fpr_b[repeat,] = tprfpr(correct_classes, predicted_classes_binnned)
        

        
    list_tpr.append(tpr)
    list_fpr.append(fpr)
    list_accuracy.append(accuracy)
    
    # list_tpr_b.append(tpr_b)
    # list_fpr_b.append(fpr_b)




x = np.linspace(0,1,100)



plt.figure()
for i in range(len(list_tpr)):
    plt.plot(np.mean(list_fpr[i]),np.mean(list_tpr[i]),'r.-')
    # plt.plot(np.mean(list_fpr_2[i]),np.mean(list_tpr_2[i]),'y.-')
plt.plot(x,x)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('$n_t$ = '+str(value_nt) +', $b$ = '+str(value_b))
plt.show()
