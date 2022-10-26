#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:11:12 2022

@author: roatisiris
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
import scipy.stats
import random

mu1, sigma1 = [0], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation

# uncomment for random sizes out of the two simulations
# size1 = random.randint(10**4, 10**6)
# size2 = random.randint(10**4, 10**6)

random.seed(1)
# uncomment for random sizes out of the two simulations
size1 = random.randint(10**4, 10**5)
size2 = np.copy(size1)

# generate from normal distrib
s1 = np.random.normal(mu1, sigma1, size1)
s2 = np.random.normal(mu2, sigma2, size2)


# see histograms for 1d normals
# if not one dimensional, comment it out
# not very informative for different population sizes
# plt.figure()
# plt.hist([s1, s2], label=['x', 'y'])
# plt.legend(loc='upper right')
# plt.show()

# create X and y to apply LDA
X = np.concatenate((s1.reshape((size1,1)),s2.reshape((size2,1))))
y = np.concatenate((np.ones((size1,)),2*np.ones((size2,))))
model = LinearDiscriminantAnalysis()
model.fit(X, y)

###########################

def lda_find_hyperplane(s1,s2,size1,size2,prob_miss_displ):
    
    # estimating mu by doing average
    estimated_mu_1 = np.array([np.mean(s1)])
    estimated_mu_2 = np.array([np.mean(s2)])

    # estimating sigma by doing sample variance
    size1min1 = size1-1
    size2min1 = size2-1
    estimated_sigma_1 = np.array([np.sum((s1 - estimated_mu_1)**2)/size1min1])
    estimated_sigma_2 = np.array([np.sum((s2 - estimated_mu_2)**2)/size2min1])

    # assumption for lda
    if prob_miss_displ:
        print('Assuming sigma1 = sigma2 which means ' + str(estimated_sigma_1)+' equal to ' +str(estimated_sigma_2))

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
        
    # calculate probability of misclassification
    if prob_miss_displ:
        Delta = np.transpose(alpha) * difference_in_est_means
        p_2given1 = scipy.stats.norm(0, 1).pdf(log_ratio_proportions/Delta - 0.5 * Delta)
        p_1given2 = scipy.stats.norm(0, 1).pdf(log_ratio_proportions/Delta - 0.5 * Delta)
        print('Prob of missclasification is '+str((p_1given2 * size2 + p_2given1 * size1)/(size1+size2)))
    
    return alpha, c
    

###########################

def lda_predict(x, alpha, c):
    if np.transpose(alpha) * x <= c:
        predicted_class = 2
    else:
        predicted_class = 1
    return predicted_class
            
    
    
###########################

alpha, c =  lda_find_hyperplane(s1,s2,size1,size2,True)

        
how_many_times_methods_agreed = 0
how_many_times_my_lda_correct = 0
how_many_times_lda_builtin_correct = 0


# start generating and checking the results of my LDA against built-in LDA

iterations = random.randint(1000, 100000)
for itera in range(iterations):
    random.seed(1)
    which_normal = random.randint(1,2)
    if which_normal == 1:
        random_simulation = np.random.normal(mu1, sigma1, (1))
    else:
        random_simulation = np.random.normal(mu2, sigma2, (1))
    predicted_class = lda_predict(random_simulation, alpha, c)
    predicted_class_builtin  = model.predict([random_simulation])
    if predicted_class_builtin == predicted_class:
        how_many_times_methods_agreed +=1
    if predicted_class_builtin == which_normal:
        how_many_times_lda_builtin_correct +=1
    if predicted_class == which_normal:
        how_many_times_my_lda_correct +=1

print('Proportion when methods agreed '+str(how_many_times_methods_agreed/iterations))
print('Proportion when my LDA correct '+str(how_many_times_my_lda_correct/iterations))
print('Proportion when built-in LDA correct '+str(how_many_times_lda_builtin_correct/iterations))


# try binning to 3 dp and then accuracy procentage