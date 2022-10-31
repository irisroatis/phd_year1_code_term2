#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:55:58 2022

@author: ir318
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


####### START OF THE CODE


# mu1 = np.array([0, 0]) # mean and standard deviation
# sigma1 = np.array([[1, 0],[ 0,1]]) # mean and standard deviation
# mu2 = np.array([1, 1]) # mean and standard deviation
# sigma2 = np.array([[1, 0],[ 0,1]]) # mean and standard deviation

mu1 = np.array([0.25]) # mean and standard deviation
sigma1 = np.array([1]) # mean and standard deviation
mu2 = np.array([1]) # mean and standard deviation
sigma2 = np.array([1]) # mean and standard deviation

dim = len(mu1)

iterations = 5000
how_many_times_repeat = 30

list_nt = [2, 5, 15, 40, 85, 200, 500]
list_b = [2, 1, 0.5,0.35, 0.2, 0.1, 0.075, 0.05, 0.0001]

# keeping track on which iteration we are on
total_number_of_cases = len(list_nt)*len(list_b)
progress = 0

###### SIMULATING DATA FOR TESTING

testing_data=[]
belonging_classes=[]

for repeat in range(how_many_times_repeat):

    random_simulation = np.zeros((iterations,dim))
    which_class_list = np.zeros((iterations,))
    
    for itera in range(iterations):

        which_normal = random.randint(1,2)
        if dim == 1:
            if which_normal == 1:
                random_simulation[itera,] = np.random.normal(mu1, sigma1)
            else:
                random_simulation[itera,] = np.random.normal(mu2, sigma2)
        else:
            if which_normal == 1:
                random_simulation[itera,] = np.random.multivariate_normal(mu1, sigma1)
            else:
                random_simulation[itera,] = np.random.multivariate_normal(mu2, sigma2)
        which_class_list[itera,] = which_normal
    
    testing_data.append(random_simulation)
    belonging_classes.append(which_class_list)

print('Testing data generated.')



###### TRAINING THE MODELS AND CHECKING IT ON SIMULATED DATA

store_proportions_lda = np.zeros((len(list_nt),len(list_b)))



for nt_index in range(len(list_nt)):
    
    nt = list_nt[nt_index]
  
    size1 = nt
    size2 = nt

    # generate from normal distrib
    if dim == 1:
        
        s1 = np.random.normal(mu1, sigma1, size1)
        s2 = np.random.normal(mu2, sigma2, size2)
        
    else:
        
        s1 = np.random.multivariate_normal(mu1, sigma1, size1)
        s2 = np.random.multivariate_normal(mu2, sigma2, size2)
            


    for bin_size_index in range(len(list_b)):

        bin_size = list_b[bin_size_index]
        
        # bins are created around 0 (one bin is set with ends in 0 and outwards
        # from there)
        bins = np.arange(0,100,bin_size)
        bins = np.concatenate((-bins[::-2], bins))

        s1_binned = put_in_bins(s1, bins)
        s2_binned = put_in_bins(s2, bins)

        ## this is if we want the built-in code
        # # create X and y to apply LDA
        # X = np.concatenate((s1_binned.reshape((size1,dim)),s2_binned.reshape((size2,dim))))
        # y = np.concatenate((np.ones((size1,)),2*np.ones((size2,))))
        # model_lda = LinearDiscriminantAnalysis()
        # model_lda.fit(X, y)
        
        # model_knn = KNeighborsClassifier(n_neighbors=5)
        # model_knn.fit(X, y)

        ## this is our own implementation of 'trainig the model' using LDA
        
        alpha, c = lda_find_hyperplane(s1_binned,s2_binned)
        # print(str(alpha) + str(c))
        tpr, fpr = list(), list()
        
        for repeat in range(how_many_times_repeat):
            
            testing_model_on_binned = put_in_bins(testing_data[repeat],bins)
            testing_model_on_unbinned = testing_data[repeat]
            correct_classes = belonging_classes[repeat]
            
            true_pos = 0
            true_neg = 0
            false_pos = 0
            false_neg = 0
            
            for itera in range(iterations):
                current_predicted_class = lda_predict(testing_model_on_unbinned[itera,], alpha, c)
                
                if correct_classes[itera,] == 2 and current_predicted_class == 1 :
                    false_pos += 1
                if correct_classes[itera,] == 1 and current_predicted_class == 1 :
                    true_pos += 1
                if correct_classes[itera,] == 1 and current_predicted_class == 2 :
                    false_neg += 1
                if correct_classes[itera,]== 2 and current_predicted_class == 2 :
                    true_neg += 1
            
            ##### THINK ABOUT WHAT TO DO WITH THESE
            
            tpr.append( true_pos / (true_pos + false_neg) )
            fpr.append (false_pos / (false_pos + false_neg) )
            
            store_proportions_lda[nt_index,bin_size_index] += (true_pos + true_neg) / iterations
      
          
        progress +=1
        print(progress/total_number_of_cases)
        
store_proportions_lda /= how_many_times_repeat 
      
plt.figure()
x,y = np.meshgrid(np.array(list_b),np.array(list_nt))
plt.scatter(x,y,c=store_proportions_lda)
plt.xlabel('bin size')
plt.ylabel('size train samples')
plt.colorbar()
plt.title('LDA ' +str(dim)+'-dimensional')
plt.show()
