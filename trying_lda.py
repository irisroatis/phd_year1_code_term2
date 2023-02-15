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

# list_bin_sizes = [0.05,0.1,0.5, 1, 1.5, 2.75, 3.3, 3.6, 4]

list_bin_sizes = [0.01,0.05,0.1,0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1, 1.25, 1.5, 2]

how_many = 200

accuracy = [np.zeros((how_many,))]
accuracy_on_binned = [np.zeros((how_many,))]

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

    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,200,bin_size)
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
        
        accuracy[i+1][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)
        accuracy_on_binned[i+1][iteration] = 1 - sum(np.absolute(my_prediction_binned - belonging_classes)) / len(belonging_classes)

dict_accuracy = {'no \n binning':accuracy[0]}
dict_accuracy_b = {'no \n binning':accuracy_on_binned[0]}

for i in range(len(list_bin_sizes)):

    dict_accuracy[str(list_bin_sizes[i])] = accuracy[i+1]
    dict_accuracy_b[str(list_bin_sizes[i])] = accuracy_on_binned[i+1]
 

# Pandas dataframe
data3 = pd.DataFrame(dict_accuracy)
# Plot the dataframe
ax = data3[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('accuracy')
plt.title('Accuracy LDA')
plt.show()

data4 = pd.DataFrame(dict_accuracy_b)
# Plot the dataframe
ax = data4[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('accuracy')
plt.title('Accuracy LDA')
plt.show()

      
            
