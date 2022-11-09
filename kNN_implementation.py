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

# uncomment for random sizes out of the two simulations
# size1 = random.randint(10**4, 10**6)
# size2 = random.randint(10**4, 10**6)

nt = 300
bin_size = 0.5

# generate from normal distrib
s1 = np.random.normal(mu1, sigma1, nt)
s2 = np.random.normal(mu2, sigma2, nt)


bins = np.arange(0,100,bin_size)
bins = np.concatenate((-bins[::-1], bins))

# bins = np.linspace(-100,100,int(200//bin_size))


s1_binned = put_in_bins(s1, bins)
s2_binned = put_in_bins(s2, bins)




# create X and y to apply LDA
X = np.concatenate((s1_binned.reshape((nt,1)),s2_binned.reshape((nt,1))))
y = np.concatenate((np.ones((nt,)),2*np.ones((nt,))))



how_many_times_repeat = 5
iterations = 100


testing_data, belonging_classes = generating_test_data(how_many_times_repeat, 
                                                        iterations, mu1, sigma1, 
                                                        mu2, sigma2, 
                                                        plot_classes = True)


list_of_thresholds = np.linspace(0, 1, 100)
list_fpr = []
list_tpr = []
list_accuracy = []



for t in list_of_thresholds:
    tpr = np.zeros([how_many_times_repeat,])
    fpr = np.zeros([how_many_times_repeat,])
    accuracy = np.zeros([how_many_times_repeat,])
    for repeat in range(how_many_times_repeat):
        predicted_classes_kNN = reg_predict(X, testing_data[repeat], y, k=71, threshold=t)
        results = tprfpr(belonging_classes[repeat], predicted_classes_kNN, accuracy = True)
        tpr[repeat] = results[0]
        fpr[repeat] = results[1]
        accuracy[repeat] = results[2]
    list_tpr.append(tpr)
    list_fpr.append(fpr)
    list_accuracy.append(accuracy)   


x = np.linspace(0,1,100)
plt.figure()
for i in range(len(list_tpr)):
    plt.plot(np.mean(list_fpr[i]),np.mean(list_tpr[i]),'r.-')
plt.plot(x,x)
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