#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 14:17:40 2023

@author: roatisiris
"""
import numpy as np
import random
import matplotlib.pyplot as plt

def generate_test(e, std, size, beta_0, beta_1):
    X = np.random.normal(e, std, size)
    y = beta_0 + beta_1 * X 
    return X, y


def st (X):
    mean = np.mean(X)
    std = np.std(X)
    norm_X =  (X - mean) / std
    return norm_X


def put_in_bins(data, bins, way_to_bin):
    digitized = np.digitize(data,bins)
    
    if way_to_bin == 'binned_centre':
        midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
        new_data = midpoints_bins[digitized-1]
    elif way_to_bin == 'binned_random':
        bin_size = bins[1] - bins[0]
        random_points_bins = bins[:len(bins)-1] + random.uniform(0, bin_size)
        new_data = random_points_bins[digitized-1]
    elif way_to_bin == 'rank':
        new_data = digitized;
    return new_data


way_to_bin = 'binned_centre'

X, y =  generate_test(5, 10, 1000, 1, 1)

bin_size = 2
bins = np.arange(0,1000,bin_size)
bins = np.concatenate((-bins[::-1][:-1], bins))

Xstd_binned = put_in_bins(st(X), bins, way_to_bin)
Xbinned_std = st(put_in_bins(X, bins, way_to_bin))

difference = Xstd_binned - Xbinned_std
print(max(difference))


print('Mean standardise then bin:', np.mean(Xstd_binned))
print('Mean bin then standardise:', np.mean(Xbinned_std))

print('Variance standardise then bin:', np.var(Xstd_binned))
print('Variance bin then standardise:', np.var(Xbinned_std))

plt.figure()
plt.plot(Xstd_binned,'.b', label = 'standardise and then bin')
plt.plot(Xbinned_std,'.r', label = 'bin and then standardise')
plt.legend()
plt.show()
