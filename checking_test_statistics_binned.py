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

mu = 0.25
sigma = 1

list_nt = [50, 100, 500, 1000, 5000, 10000, 50000]
bin_size = 2
bins = np.arange(0,100,bin_size)
bins = np.concatenate((-bins[::-2], bins))
how_many = 100
list_mean = []
list_mean_binned = []
list_difference = []

for nt in list_nt:
    sample_mean = np.zeros((how_many,))
    sample_mean_binned = np.zeros((how_many,))
    for iterations in range(how_many):
        random_simulation_from_normal =  np.random.normal(mu, sigma, nt)
        binned_simulted_data = put_in_bins(random_simulation_from_normal, bins)
        sample_mean[iterations,] = np.mean(random_simulation_from_normal)
        sample_mean_binned[iterations,] = np.mean(binned_simulted_data)
    list_mean.append(sample_mean)
    list_mean_binned.append(sample_mean_binned)
    list_difference.append(sample_mean_binned-mu)
    
fig, ax = plt.subplots()
ax.boxplot(list_mean)
ax.set_xticklabels(list_nt)
plt.title('Boxplots of estimated means from unbinned data')
plt.axhline(mu,color = 'g',linestyle = '--',linewidth = 1)
plt.show()

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
plt.title('Boxplots of estimated means from binned data')
plt.show()