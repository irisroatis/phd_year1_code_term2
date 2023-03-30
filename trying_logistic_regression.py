#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 14:02:44 2023

@author: roatisiris
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.linear_model import LogisticRegression
import random


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

            which_normal = random.randint(0,1)
            if dim == 1:
                if which_normal == 0:
                    random_simulation[itera,] = np.random.normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.normal(mu2, sigma2)
            else:
                if which_normal == 0:
                    random_simulation[itera,] = np.random.multivariate_normal(mu1, sigma1)
                else:
                    random_simulation[itera,] = np.random.multivariate_normal(mu2, sigma2)
            which_class_list[itera,] = which_normal
        
        testing_data.append(random_simulation)
        belonging_classes.append(which_class_list)
      
    
    return testing_data, belonging_classes

def predict_logistic(X_test, beta, beta_0, threshold):
    n = X_test.shape[0]
    y_pred = np.zeros((n,))
    # Compute vector y_log predicting the probabilities
    y_log = logistic(X_test @ beta + beta_0) 
    for i in range(y_log.shape[0]):
        # Convert probabilities y_log to actual predictions y_pred using the threshold
        if y_log[i] > threshold:
            y_pred[i] = 1  
        else:
            y_pred[i] = 0  
    return y_pred

def logistic(x):
    logistic =  1/(1+np.exp(-x))
    return logistic

def put_in_bins(data, bins, ranking = False):
    digitized = np.digitize(data,bins)
    if not ranking:
        midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
        new_data = midpoints_bins[digitized-1]
        return new_data
    else:
        return digitized

size = 1000

# list_bin_sizes = [0.05,0.1,0.5, 1, 1.5, 2.75, 3.3, 3.6, 4]

list_bin_sizes = [0.01,0.05,0.1,0.25, 0.35, 0.5, 0.65, 0.75, 0.9, 1, 1.25, 1.5, 2]

how_many = 100

list_error_beta0 = [np.zeros((how_many,))]
list_error_beta1 = [np.zeros((how_many,))]
accuracy = [np.zeros((how_many,))]
accuracy_shepp = [np.zeros((how_many,))]

mu1, sigma1 = [0.25], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation


test_data, belonging_classes = generating_test_data(1, 10000, mu1, sigma1, mu2, 
                         sigma2, plot_classes = False)
test_data = test_data[0]
belonging_classes = belonging_classes[0]

list_of_bins = []
for i in range(len(list_bin_sizes)):
    list_error_beta0.append(np.zeros((how_many,)))
    list_error_beta1.append(np.zeros((how_many,)))
    accuracy.append(np.zeros((how_many,)))
    accuracy_shepp.append(np.zeros((how_many,)))
  
    bin_size = list_bin_sizes[i]
    bins = np.arange(bin_size/2,200,bin_size)
    bins = np.concatenate((-bins[::-1], bins))
    list_of_bins.append(bins)
    


for iteration in range(how_many):
    
    s1 = np.random.normal(mu1, sigma1, size)
    s2 = np.random.normal(mu2, sigma2, size)
    
    
    x = np.concatenate((s1.reshape((size,)),s2.reshape((size,))))
    y = np.concatenate((0*np.ones((size,)),np.ones((size,))))
    
    
    model = LogisticRegression(penalty='none').fit(x.reshape(-1,1), y)

    list_error_beta0[0][iteration] = model.intercept_ 
    list_error_beta1[0][iteration] = model.coef_[0]
    
    
    my_prediction = model.predict(test_data)
    accuracy[0][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)
    accuracy_shepp[0][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)
   
    for i in range(len(list_bin_sizes)):
        bins = list_of_bins[i] 
        new_X = put_in_bins(x, bins, ranking = True)
        
        variance_sample = np.var(new_X)
     
        model = LogisticRegression(penalty='none').fit(new_X.reshape(-1,1), y)

        list_error_beta0[i+1][iteration] = model.intercept_ 
        list_error_beta1[i+1][iteration] = model.coef_[0] 
        
        corrected_gradient = model.coef_[0] / (1 - list_bin_sizes[i]**2 / (12 * variance_sample))
        
        my_prediction = model.predict(test_data)
        accuracy[i+1][iteration] = 1 - sum(np.absolute(my_prediction - belonging_classes)) / len(belonging_classes)
        prediction_shepp = predict_logistic(test_data, corrected_gradient, model.intercept_ , 0.5)
        accuracy_shepp[i+1][iteration] = 1 - sum(np.absolute(prediction_shepp - belonging_classes)) / len(belonging_classes)
        
dictionary_beta0 = {'no \n binning':list_error_beta0[0]}
dictionary_beta1 = {'no \n binning':list_error_beta1[0]}
dict_accuracy = {'no \n binning':accuracy[0]}
dict_accuracy_shepp = {'no \n binning':accuracy_shepp[0]}

for i in range(len(list_bin_sizes)):
    dictionary_beta0[str(list_bin_sizes[i])] = list_error_beta0[i+1]
    dictionary_beta1[str(list_bin_sizes[i])] = list_error_beta1[i+1]
    dict_accuracy[str(list_bin_sizes[i])] = accuracy[i+1]
    dict_accuracy_shepp[str(list_bin_sizes[i])] = accuracy_shepp[i+1]
  

# Pandas dataframe
data1 = pd.DataFrame(dictionary_beta1)
# Plot the dataframe

ax = data1[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('$\\hat{\\beta_1^{*}}$')
plt.title('Estimated Gradient in Logistic Regression')
plt.show()

# Pandas dataframe
data2 = pd.DataFrame(dictionary_beta0)
# Plot the dataframe
ax = data2[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('$\\hat{\\beta}^{*} - \\beta$')
plt.xlabel('bin size, $h$')
plt.ylabel(' $\\hat{\\beta_0^{*}}$')
plt.title('Estimated Intercept in Logistic Regression')
plt.show()

# Pandas dataframe
data3 = pd.DataFrame(dict_accuracy)
# Plot the dataframe
ax = data3[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('accuracy')
plt.title('Accuracy Logistic Regression')
plt.show()

# Pandas dataframe
data4 = pd.DataFrame(dict_accuracy_shepp)
# Plot the dataframe
ax = data4[['no \n binning','0.01','0.05','0.1','0.25', '0.35', '0.5', '0.65', '0.75', '0.9', '1', '1.25', '1.5','2']].plot(kind='box', title='boxplot')
plt.xlabel('bin size, $h$')
plt.ylabel('accuracy')
plt.title('Accuracy Logistic Regression where $\\hat{\\beta_1^{*}}$ was Shppard Corrected')
plt.show()
