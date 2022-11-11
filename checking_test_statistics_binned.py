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

mu = 1
sigma = 1

list_nt = [5, 50, 100, 500, 1000, 5000, 10000, 50000]
bin_size = 0.01
bins = np.arange(bin_size/2,100,bin_size)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))
how_many = 100
list_mean_binned = []
list_difference = []
list_sample_variance = []

for nt in list_nt:
    sample_mean_binned = np.zeros((how_many,))
    sample_variance = np.zeros((how_many,))
    for iterations in range(how_many):
        random_simulation_from_normal =  np.random.normal(mu, sigma, nt)

        binned_simulted_data = put_in_bins(random_simulation_from_normal, bins)
        sample_mean_binned[iterations,] = np.mean(binned_simulted_data)
        sample_variance[iterations,] = sum((binned_simulted_data -  sample_mean_binned[iterations,])**2) / (nt-1) 
    list_mean_binned.append(sample_mean_binned)
    list_difference.append(sample_mean_binned-mu)
    list_sample_variance.append(sample_variance)
    
fig, ax = plt.subplots()
ax.boxplot(list_mean_binned)
ax.set_xticklabels(list_nt)
plt.axhline(mu,color = 'g',linestyle = '--',linewidth = 1)
plt.title('Boxplots of estimated means from binned data')
plt.show()

fig, ax = plt.subplots()
ax.boxplot(list_difference)
ax.set_xticklabels(list_nt)
plt.axhline(0,color = 'g',linestyle = '--',linewidth = 1)
plt.title('Boxplots of bias of mean')
plt.show()

fig, ax = plt.subplots()
ax.boxplot(list_sample_variance)
ax.set_xticklabels(list_nt)
plt.axhline(sigma + bin_size**2/12,color = 'g',linestyle = '--',linewidth = 1)
plt.title('Boxplots of sample variances about the estimated mean')
plt.show()