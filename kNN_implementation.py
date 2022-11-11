#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 11:02:47 2022

@author: ir318
"""

import numpy as np
from defined_functions import *
import matplotlib.pyplot as plt

mu1, sigma1 = [0], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation


how_many_times_repeat = 1
iterations = 1000
list_of_thresholds = np.linspace(0, 1, 50)

#generating test data
testing_data, belonging_classes = generating_test_data(how_many_times_repeat, 
                                                        iterations, mu1, sigma1, 
                                                        mu2, sigma2, 
                                                        plot_classes = False)

###### First Set of Parameters
nt1 = 400
bin_size1 = 0.01
k1 = 71
bins = np.arange(bin_size1/2,100,bin_size1)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))

s1 = np.random.normal(mu1, sigma1, nt1)
s2 = np.random.normal(mu2, sigma2, nt1)
s1_binned = put_in_bins(s1, bins) #binned
s2_binned = put_in_bins(s2, bins) #binned
# create X and y to apply kNN
X = np.concatenate((s1_binned.reshape((nt1,1)),s2_binned.reshape((nt1,1))))
y = np.concatenate((np.ones((nt1,)),2*np.ones((nt1,))))
list_tpr, list_fpr, list_accuracy = roc_kNN(X, y, testing_data, belonging_classes, list_of_thresholds, how_many_times_repeat, k1)



###### Second Set of Parameters
nt2 = 400
bin_size2 = 0.01
k2 = 5
bins = np.arange(bin_size2/2,100,bin_size2)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))

s1_2 = np.random.normal(mu1, sigma1, nt2)
s2_2 = np.random.normal(mu2, sigma2, nt2)
s1_2_binned = put_in_bins(s1_2, bins) #binned
s2_2_binned = put_in_bins(s2_2, bins) #binned
# create X and y to apply kNN
X2 = np.concatenate((s1_2_binned.reshape((nt2,1)),s2_2_binned.reshape((nt2,1))))
y2 = np.concatenate((np.ones((nt2,)),2*np.ones((nt2,))))
list_tpr2, list_fpr2, list_accuracy2 = roc_kNN(X2, y2, testing_data, belonging_classes, list_of_thresholds, how_many_times_repeat, k2)




x = np.linspace(0,1,100)
plt.figure()

plt.plot(list_fpr,list_tpr,'r.-',label='$n_t=$'+str(nt1)+',$b=$'+str(bin_size1)+',$k=$'+str(k1))
plt.plot(list_fpr2,list_tpr2,'y.-',label='$n_t=$'+str(nt2)+',$b=$'+str(bin_size2)+',$k=$'+str(k2))
plt.plot(x,x)
plt.legend()
plt.xlabel('fpr')
plt.ylabel('tpr')
# plt.title('$n_t$ = '+str(value_nt) +', $b$ = '+str(value_b))
plt.show()


#### TRYING BUILT-IN ONE

# from sklearn.neighbors import KNeighborsClassifier
# regressor = KNeighborsClassifier(n_neighbors=5)
# regressor.fit(X, y)

# predicted_kNN_builtin = regressor.predict(testing_data[0]).tolist()

# print(predicted_kNN_builtin == predicted_classes_kNN)

