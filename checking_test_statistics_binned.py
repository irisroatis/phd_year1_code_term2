#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:16:26 2022

@author: roatisiris
"""
import random
import numpy as np
from defined_functions import *
import matplotlib.pyplot as plt

mu = 0
sigma = 1

list_nt = [5, 10, 50, 100, 500, 1000, 5000, 10000]
bin_size = 0.5
bins = np.arange(bin_size/2,100,bin_size)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))
how_many = 100
list_mean_binned = []
list_sample_variance = []

for nt in list_nt:
    difference_sample_mean = np.zeros((how_many,))
    difference_sample_variance = np.zeros((how_many,))
    for iterations in range(how_many):
        random_simulation_from_normal =  np.random.normal(mu, sigma, nt)
        binned_simulted_data = put_in_bins(random_simulation_from_normal, bins)
        mean_binned =  np.mean(binned_simulted_data)
        mean_unbinned =  np.mean(random_simulation_from_normal) 
        
        difference_sample_mean[iterations,] = mean_binned - mean_unbinned
        difference_sample_variance[iterations,] = sum((binned_simulted_data -  mean_binned)**2) / (nt-1) - sum((random_simulation_from_normal -  mean_unbinned)**2) / (nt-1)
    list_mean_binned.append(difference_sample_mean)
    list_sample_variance.append(difference_sample_variance)
    
fig, ax = plt.subplots()
ax.boxplot(list_mean_binned)
ax.set_xticklabels(list_nt)
plt.axhline(0,color = 'g',linestyle = '--',linewidth = 1, label = '0')
plt.xlabel('size of sample')
plt.ylabel('$E[X] - E[X^{*}]$')
plt.title('Boxplots of Difference in Mean of Unbinned Data \n vs Binned Data')
plt.legend()
plt.show()

fig, ax = plt.subplots()
ax.boxplot(list_sample_variance)
ax.set_xticklabels(list_nt)
plt.axhline(bin_size**2/12,color = 'g',linestyle = '--',linewidth = 1, label = '$\\frac{h^2}{12}$')
plt.xlabel('size of sample')
plt.ylabel('$V[X] - V[X^{*}]$')
plt.title('Boxplots of Difference in Variance of Unbinned Data \n about the Mean vs Binned Data')
plt.legend()
plt.show()