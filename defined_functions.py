#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:20:40 2022

@author: roatisiris
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier 
import scipy.stats
from mpl_toolkits import mplot3d
import random

####### FUNCTIONS NEEDED

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data



def lda_find_hyperplane(s1,s2):
    
    size1 = len(s1)
    size2 = len(s2)
    
    # estimating mu by doing average
    estimated_mu_1 = np.array([np.mean(s1)])
    estimated_mu_2 = np.array([np.mean(s2)])

    # estimating sigma by doing sample variance
    size1min1 = size1-1
    size2min1 = size2-1
    estimated_sigma_1 = np.array([np.sum((s1 - estimated_mu_1)**2)/size1min1])
    estimated_sigma_2 = np.array([np.sum((s2 - estimated_mu_2)**2)/size2min1]) 

    sigma = 1/ (size1+ size2 - 2) *(size1min1 * estimated_sigma_1 + size2min1 * estimated_sigma_2)

    log_ratio_proportions = np.log(size2 / size1)
    difference_in_est_means = estimated_mu_1 - estimated_mu_2


    # define the alpha (normal to the boundary plane)
    if sigma.shape != (1,):
        alpha = np.linalg.inv(sigma) * difference_in_est_means
        c = 0.5 * np.traspose(alpha) * (estimated_mu_1 + estimated_mu_2) + log_ratio_proportions
    else:
        alpha = 1/sigma * difference_in_est_means
        c = 0.5 * alpha * (estimated_mu_1 + estimated_mu_2) + log_ratio_proportions
    
    return alpha, c
    

def lda_predict(x, alpha, c):
    if np.transpose(alpha) * x <= c:
        predicted_class = 2
    else:
        predicted_class = 1
    return predicted_class

