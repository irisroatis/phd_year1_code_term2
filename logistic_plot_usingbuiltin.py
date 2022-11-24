#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:09:12 2022

@author: roatisiris
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
import random
import matplotlib.pyplot as plt

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
    

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data


def predict_logistic(probabilities,threshold):
    n = probabilities.shape[0]
    y_pred = np.zeros((n,1))
    # Compute vector y_log predicting the probabilities
    for i in range(n):
        # Convert probabilities y_log to actual predictions y_pred using the threshold
        if probabilities[i,0] > threshold:
            y_pred[i,0] = 1  
        else:
            y_pred[i,0] = 0  
    return y_pred



def tprfpr_log(actual, predicted, accuracy = False):
    falseneg, falsepos, truepos, trueneg = 0, 0, 0, 0
    for i in range(len(actual)):
        if actual[i] == 0 and predicted[i] == 1 :
            falsepos += 1
        if actual[i] == 1 and predicted[i] == 1 :
            truepos += 1
        if actual[i] == 1 and predicted[i] == 0 :
            falseneg += 1
        if actual[i] == 0 and predicted[i] == 0 :
            trueneg += 1
    tpr = truepos / (truepos + falseneg)
    fpr = falsepos / (falsepos + trueneg)
    if accuracy:
        return  tpr, fpr, (truepos+trueneg) / (truepos+trueneg+falsepos+falseneg)
    else:
        return tpr, fpr


def roc_logistic(X, y, testing_data, belonging_classes, list_of_thresholds, how_many_times_repeat):
    
    list_fpr = []
    list_tpr = []
    list_accuracy = []

    for t in list_of_thresholds:
        tpr = 0
        fpr = 0
        accuracy = 0
        for repeat in range(how_many_times_repeat):
            clf = LogisticRegression(penalty='none').fit(X, y)
            predicted_classes_probabilities =  clf.predict_proba(testing_data[repeat])
            predicted_classes = predict_logistic(predicted_classes_probabilities,t)
            results = tprfpr_log(belonging_classes[repeat], predicted_classes, accuracy = True)
            tpr += results[0]
            fpr += results[1]
            accuracy += results[2]
        tpr /= how_many_times_repeat
        fpr /= how_many_times_repeat
        accuracy /= how_many_times_repeat
        
        list_tpr.append(tpr)
        list_fpr.append(fpr)
        list_accuracy.append(accuracy)   
    
    return list_tpr, list_fpr, list_accuracy
    

mu1, sigma1 = [0.25], [1] # mean and standard deviation
mu2, sigma2 = [1.25], [1] # mean and standard deviation


how_many_times_repeat = 1
iterations = 1000
list_of_thresholds = np.linspace(0, 1, 50)

#generating test data
testing_data, belonging_classes = generating_test_data(how_many_times_repeat, 
                                                        iterations, mu1, sigma1, 
                                                        mu2, sigma2, 
                                                        plot_classes = False)

###### First Set of Parameters
nt1 = 400
bin_size1 = 2
bins = np.arange(bin_size1/2,100,bin_size1)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))

s1 = np.random.normal(mu1, sigma1, nt1)
s2 = np.random.normal(mu2, sigma2, nt1)
s1_binned = put_in_bins(s1, bins) #binned
s2_binned = put_in_bins(s2, bins) #binned

# create X and y to apply classification
X = np.concatenate((s1.reshape((nt1,1)),s2.reshape((nt1,1))))
y = np.concatenate((0*np.ones((nt1,)),np.ones((nt1,))))

# # initialising the parameters
# beta = np.zeros((X.shape[1], 1))
# beta_0 = 0
# num_iterations = 100
# parameters= optimise(X, y, beta, beta_0, num_iterations)
    
# beta_not_binned = parameters["beta"]
# beta_0_not_binned = parameters["beta_0"]

#### same thing for binned data

X_binned = np.concatenate((s1_binned.reshape((nt1,1)),s2_binned.reshape((nt1,1))))
y_binned = np.concatenate((0*np.ones((nt1,)),np.ones((nt1,))))


# parameters= optimise(X_binned, y_binned, beta, beta_0, num_iterations)
    
# beta_binned = parameters["beta"]
# beta_0_binned = parameters["beta_0"]

# y_pred_not_binned = predict_logistic(testing_data[0], beta_not_binned, beta_0_not_binned, 0.5)
# y_pred_binned = predict_logistic(testing_data[0], beta_binned, beta_0_binned, 0.5)




#### ROC curves
list_tpr, list_fpr, list_accuracy = roc_logistic(X, y, testing_data, belonging_classes, list_of_thresholds, how_many_times_repeat)

list_tpr_b, list_fpr_b, list_accuracy_b = roc_logistic(X_binned, y_binned, testing_data, belonging_classes, list_of_thresholds, how_many_times_repeat)

x = np.linspace(0,1,100)
plt.figure()

plt.plot(list_fpr,list_tpr,'r.')
plt.plot(list_fpr_b,list_tpr_b,'y.')
plt.plot(x,x)
plt.xlabel('fpr')
plt.ylabel('tpr')
# plt.title('$n_t$ = '+str(value_nt) +', $b$ = '+str(value_b))
plt.show()