#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 10:58:36 2023

@author: roatisiris
"""

import numpy as np

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data

def right_or_left_bins(data, bins, left):
    digitized = np.digitize(data,bins)
    left_bins = bins[:len(bins)-1] 
    right_bins = bins[1:]
    if left:
        return left_bins[digitized-1]
    else:
        return right_bins[digitized-1]

def create_bins(bin_size):
    how_many = 100
    bins = np.arange(bin_size/2,how_many,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    return bins

def param_regression(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def add_col_ones(X):
    ones = np.ones((X.shape[0],))
    return np.vstack((ones, X)).T
    

beta_0_true = 0.4
beta_1_true = 0.6

size = 1000

X = np.random.normal(5, 5, size)
y = beta_0_true + beta_1_true * X + np.random.normal(0, 1, size)

X_test = np.random.normal(5, 5, 100)
y_test = beta_0_true + beta_1_true * X_test + np.random.normal(0, 1, 100)

bins_X = create_bins(1)
bins_y = create_bins(0.5)

X_binned = put_in_bins(X, bins_X)
y_binned = put_in_bins(y, bins_y)

beta_binned = param_regression(add_col_ones(X_binned), y_binned)

X_l = right_or_left_bins(X, bins_X, True)
X_r = right_or_left_bins(X, bins_X, False)

y_l = right_or_left_bins(y, bins_y, True)
y_r = right_or_left_bins(y, bins_y, False)

beta_r = param_regression(add_col_ones(X_r - X_l), y_r - y_l)



