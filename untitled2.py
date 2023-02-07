#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:42:13 2023

@author: roatisiris
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data

size = 100000

bin_size = 1
bins = np.arange(bin_size/2,200,bin_size)
bins = np.concatenate((-bins[::-1], bins))


#list_bin_sizes = [0.01,0.05,0.1,0.5, 0.75, 1, 1.5, 2.25, 2.75, 3.3, 3.6, 4]

how_many = 100

list_error_beta0 = [np.zeros((how_many,))]
list_error_beta1 = [np.zeros((how_many,))]

mu1, sigma1 = [0.25], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation

s1 = np.random.normal(mu1, sigma1, size)
s2 = np.random.normal(mu2, sigma2, size)

x = np.concatenate((s1.reshape((size,)),s2.reshape((size,))))

x_binned = put_in_bins(x, bins)

epsilon = x_binned - x

plt.plot(x, epsilon,'.')
plt.xlabel('x')
plt.ylabel('$\epsilon$')
plt.show()

plt.plot(x_binned, epsilon, '.')
plt.xlabel('$x^*$')
plt.ylabel('$\epsilon$')
plt.show()

plt.hist(x_binned)
plt.title('hist $x^*$')
plt.show()

plt.hist(epsilon)
plt.title('hist $\epsilon$')
plt.show()
print(np.corrcoef(x, epsilon))
