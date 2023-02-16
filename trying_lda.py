#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:23:05 2023

@author: roatisiris
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import random
from defined_functions import *


size = 1000

list_bin_sizes = np.linspace(0.01,5,100)

# list_bin_sizes = [0.01,0.05,0.1,0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1, 1.25, 1.5, 2]

how_many = 100

accuracy = [np.zeros((how_many,))]
accuracy_on_binned = [np.zeros((how_many,))]
store_maha_dist = []
store_bhatt_dist = []


mu1, sigma1 = [0.5], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation


test_data, belonging_classes = generating_test_data(1, 10000, mu1, sigma1, mu2, 
                         sigma2, plot_classes = False)
test_data = test_data[0]
belonging_classes = belonging_classes[0]



list_of_bins = []
for i in range(len(list_bin_sizes)):
  
    accuracy.append(np.zeros((how_many,)))
    accuracy_on_binned.append(np.zeros((how_many,)))
    store_maha_dist.append(np.zeros((how_many,)))
    store_bhatt_dist.append(np.zeros((how_many,)))

    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,1000,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    list_of_bins.append(bins)
    


for iteration in range(how_many):
    
    s1 = np.random.normal(mu1, sigma1, size)
    s2 = np.random.normal(mu2, sigma2, size)
    
    x = np.concatenate((s1.reshape((size,)),s2.reshape((size,))))
    y = np.concatenate((0*np.ones((size,)),np.ones((size,))))
    
    model = LinearDiscriminantAnalysis().fit(x.reshape(-1,1),y)

    
    my_prediction = model.predict(test_data)
    accuracy[0][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)

    accuracy_on_binned[0][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)

    
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(x, bins)
     
        model = LinearDiscriminantAnalysis().fit(new_X.reshape(-1,1), y)

        my_prediction = model.predict(test_data)
        
        binned_test_data = put_in_bins(test_data, bins)
        my_prediction_binned = model.predict(binned_test_data)
        
        store_maha_dist[i][iteration] = maha_distance(x, new_X)
        store_bhatt_dist[i][iteration] = bhatt_distance(x, new_X)
        
        accuracy[i+1][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)
        accuracy_on_binned[i+1][iteration] = 1 - sum(np.absolute(my_prediction_binned - belonging_classes)) / len(belonging_classes)

dict_accuracy = {'no \n binning':accuracy[0]}
dict_accuracy_b = {'no \n binning':accuracy_on_binned[0]}
dict_accuracy_mean = [np.mean(accuracy[0])]
dict_accuracy_b_mean = [np.mean(accuracy_on_binned[0])]
dict_maha_dist = {}
dict_bhatt_dist = {}
dict_maha_dist_mean = []
dict_bhatt_dist_mean = []


for i in range(len(list_bin_sizes)):

    dict_accuracy[str(list_bin_sizes[i])] = accuracy[i+1]
    dict_accuracy_b[str(list_bin_sizes[i])] = accuracy_on_binned[i+1]
    dict_accuracy_mean.append(np.mean(accuracy[i+1]))
    dict_accuracy_b_mean.append(np.mean(accuracy_on_binned[i+1]))
    dict_maha_dist[str(list_bin_sizes[i])] = store_maha_dist[i]
    dict_bhatt_dist[str(list_bin_sizes[i])] = store_bhatt_dist[i]
    dict_maha_dist_mean.append(np.mean(store_maha_dist[i]))
    dict_bhatt_dist_mean.append(np.mean(store_bhatt_dist[i]))
 

# # Pandas dataframe
# data1 = pd.DataFrame(dict_accuracy)
# # Plot the dataframe
# ax = data1[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
# plt.xlabel('bin size, $h$')
# plt.ylabel('accuracy')
# plt.title('Accuracy LDA')
# plt.show()

# data2 = pd.DataFrame(dict_accuracy_b)
# # Plot the dataframe
# ax = data2[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
# plt.xlabel('bin size, $h$')
# plt.ylabel('accuracy')
# plt.title('Accuracy LDA')
# plt.show()


# data3 = pd.DataFrame(dict_maha_dist)
# # Plot the dataframe
# ax = data3[['0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
# plt.xlabel('bin size, $h$')
# plt.ylabel('$D_{ma}(p,q)$')
# plt.title('Mahalanobis Distance')
# plt.show()

# data4 = pd.DataFrame(dict_bhatt_dist)
# # Plot the dataframe
# ax = data4[['0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
# plt.xlabel('bin size, $h$')
# plt.ylabel('$D_{bh}(p,q)$')
# plt.title('Bhattacharya Distance')
# plt.show()

plt.scatter(list(list_bin_sizes), dict_maha_dist_mean)
plt.title('Mahalanobis Distance')
plt.xlabel('bin size, $h$')
plt.ylabel('$E[D_{ma}(p,q)]$')
plt.show()

plt.scatter(list(list_bin_sizes), dict_bhatt_dist_mean)
plt.title('Bhattacharya Distance')
plt.xlabel('bin size, $h$')
plt.ylabel('$E[D_{bh}(p,q)]$')
plt.show()

# plt.scatter([0] + list(list_bin_sizes), dict_accuracy_mean)
# plt.title('Non-binned Test Data')
# plt.xlabel('bin size, $h$')
# plt.ylabel('$E[$Accuracy$]$')
# plt.show()

# plt.scatter([0] + list(list_bin_sizes), dict_accuracy_b_mean)
# plt.title('Binned Test Data')
# plt.xlabel('bin size, $h$')
# plt.ylabel('$E[$Accuracy$]$')
# plt.show()

plt.scatter([0] + dict_bhatt_dist_mean, dict_accuracy_b_mean)
plt.title('Binned Test Data, Bhattacharya')
plt.ylabel('$E[$Accuracy$]$')
plt.xlabel('E[D_{bh}(p,q)]')
plt.show()

plt.scatter([0] + dict_maha_dist_mean, dict_accuracy_b_mean)
plt.title('Binned Test Data, Mahalanobis')
plt.ylabel('$E[$Accuracy$]$')
plt.xlabel('E[D_{ma}(p,q)]')
plt.show()





      
            
