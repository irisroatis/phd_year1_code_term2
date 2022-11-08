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
    
    if sigma == 0:
        raise TypeError("Variance is 0. Check that the bins are not too large.")

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


def tprfpr(actual, predicted, accuracy = False):
    falseneg, falsepos, truepos, trueneg = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 2 and predicted[i] ==1 :
            falsepos += 1
        if actual[i] == 1 and predicted[i] == 1 :
            truepos += 1
        if actual[i] == 1 and predicted[i] == 2 :
            falseneg += 1
        if actual[i] == 2 and predicted[i] == 2 :
            trueneg += 1
    tpr = truepos / (truepos + falseneg)
    fpr = falsepos / (falsepos + trueneg)
    if accuracy:
        return  tpr, fpr, (truepos+trueneg) / (truepos+trueneg+falsepos+falseneg)
    else:
        return tpr, fpr

def generating_test_data(how_many_times_repeat, iterations, mu1, sigma1, mu2, 
                         sigma2, plot_classes = False):
    """

    Parameters
    ----------
    how_many_times_repeat : how many different samples wanted (how many
                                                               experiments)
    iterations : sample size (how many realisations in a single experiment)
    mu1 : mean of first Normal
    sigma1 : variance of first Normal
    mu2 : mean of second Normal
    sigma2 : variance of second Normal

    Returns
    -------
    testing_data : a list of arrays. Each array is an experiment who has 
                   multiple realisations from both Normals
    belonging_classes : the true classes of the samples from the experiments
                        (the true normal each realisation comes from)

    """
    
    dim = len(mu1)
    testing_data=[]
    belonging_classes=[]

    for repeat in range(how_many_times_repeat):

        random_simulation = np.zeros((iterations,dim))
        which_class_list = np.zeros((iterations,))
        
        for itera in range(iterations):

            which_normal = random.randint(1,2)
            if dim == 1:
                if which_normal == 1:
                    random_simulation[itera,] = np.random.normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.normal(mu2, sigma2)
            else:
                if which_normal == 1:
                    random_simulation[itera,] = np.random.multivariate_normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.multivariate_normal(mu2, sigma2)
            which_class_list[itera,] = which_normal
        
        testing_data.append(random_simulation)
        belonging_classes.append(which_class_list)
        
        if plot_classes:
            trying_sth = testing_data[repeat]
            plt.hist(trying_sth[belonging_classes[repeat] == 1], alpha=0.5, label='N('+str(mu1[0])+','+str(sigma1[0])+')')
            plt.hist(trying_sth[belonging_classes[repeat] == 2], alpha=0.5, label='N('+str(mu2[0])+','+str(sigma2[0])+')')
            plt.legend(loc='upper right')
            plt.title('testing data')
            plt.show()
    
    return testing_data, belonging_classes
    

####### FUNCTIONS FOR kNN


def euclidian_distance(p, q):
    '''
    
    Parameters
    ----------
    p, q : D x 1 vectors
        Two points in our D-dimensional Euclidean Space

    Returns
    -------
    eucl : float
        It outputs the Euclidean Distance d(p,q).

    '''
    eucl = np.sqrt(np.sum((p-q)**2, axis=1))
    return eucl


def k_neighbours(X_train, X_test, k, return_distance=False):
    '''
     Parameters
    ----------
    X_train: N x M matrix
        The dataset of points which are the set of the neighbours.
    X_test : K x D matrix
        The dataset of points for which the neighbours will be found.
    k : integer
        The number of neighbours.
    return_distance : boolean
        Return or not the distance.

    Returns
    -------
    neigh_ind : list
        List containing the indices of the k nearest neighbours.

    '''
    n_neighbours = k
    dist = []
    neigh_ind = []
  
    # compute distance from each point x_text in X_test to all points in X_train (hint: use python's list comprehension)
    point_dist = [euclidian_distance(x_test, X_train) for x_test in X_test] 

    # determine which k training points are closest to each test point
    for row in point_dist:
        enum_neigh = enumerate(row)
        sorted_neigh = sorted(enum_neigh, key=lambda x: x[1])[:k]

        ind_list = [tup[0] for tup in sorted_neigh]
        dist_list = [tup[1] for tup in sorted_neigh]

        dist.append(dist_list)
        neigh_ind.append(ind_list)
  
    # return distances together with indices of k nearest neighbouts
    if return_distance:
        return np.array(dist), np.array(neigh_ind)
  
    return neigh_ind


def reg_predict(X_train, X_test, y_train, k):
    '''
     Parameters
    ----------
    X_train: N x M matrix
        The dataset of points which are the set of the neighbours.
    X_test : K x D matrix
        The dataset of points for which we predict the target variables.
    y_train : N x 1 vector
        The target variable for the test set.
    k : integer
        Number of neighbours.

    Returns
    -------
    y_pred : K x 1 vector
        Predicted target variables for test data.

    '''
    # each of the k neighbours contributes equally to the classification of any data point in X_test  
    neighbours = np.array(k_neighbours(X_train, X_test, k))
    # compute mean over neighbours labels 
    y_pred = np.array([np.mean(y_train[neighbour]) for neighbour in neighbours]) 
    return y_pred



