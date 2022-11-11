#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 10:47:31 2022

@author: ir318
"""
import numpy as np

def logistic(x):
    logistic =  1/(1+np.exp(-x))
    return logistic

def predict_log(X, beta, beta_0):
    y_log = logistic(beta.T @ X + beta_0) # Use formula above
    return y_log

def initialise(d):
    beta = np.zeros(shape=(d, 1), dtype=np.float32) # Initialise beta
    beta_0 = 0 # Initialise beta_0
    return beta, beta_0

def propagate(X, y, beta, beta_0):
    n = X.shape[1] 
    y_log = predict_log(X, beta, beta_0) # Find predictions
    # Find cost function
    cost = (-1) * np.mean(np.multiply(y,np.log(y_log)) + np.multiply(1-y,np.log(1-y_log)), axis = 1) 
    cost = np.squeeze(cost)
    # Find derivatives
    dbeta = (1/n) * X @ np.transpose(y_log - y)
    dbeta_0 = np.mean(y_log-y)
    grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}  # Store gradients in a dictionary
    return grads, cost



def optimise(X, y, beta, beta_0, num_iterations, learning_rate=0.005):
    costs = list() # Initialise empty list of costs 
    for i in range(num_iterations):
        # Calculate cost and gradients 
        grads, cost = propagate(X, y, beta, beta_0)
        # Retrieve derivatives from dictionary grads
        dbeta = grads["dbeta"]
        dbeta_0 = grads["dbeta_0"]  
        # Updating procedure for the parameters and offset
        beta -= learning_rate * dbeta
        beta_0 -= learning_rate * dbeta_0 
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
    # Save parameters and gradients in dictionary
    params = {"beta": beta, "beta_0": beta_0}
    grads = {"dbeta": dbeta, "dbeta_0": dbeta_0}
    return params, grads, costs

def predict_logistic(X_test, beta, beta_0, threshold):
    n = X_test.shape[1]
    y_pred = np.zeros((1,n))
    beta = beta.reshape(X_test.shape[0], 1)
    # Compute vector y_log predicting the probabilities
    y_log = predict_log(X_test, beta, beta_0)
    for i in range(y_log.shape[1]):
        # Convert probabilities y_log to actual predictions y_pred using the threshold
        if y_log[0, i] > threshold:
            y_pred[0, i] = 1  
        else:
            y_pred[0, i] = 0  
    return y_pred


mu1, sigma1 = [0], [3] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation


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
bin_size1 = 0.01
k1 = 5
bins = np.arange(bin_size1/2,100,bin_size1)
bins = np.concatenate((-bins[::-1], bins))
# bins = np.linspace(-100,100,int(200//bin_size))

s1 = np.random.normal(mu1, sigma1, nt1)
s2 = np.random.normal(mu2, sigma2, nt1)
s1_binned = put_in_bins(s1, bins) #binned
s2_binned = put_in_bins(s2, bins) #binned
# create X and y to apply kNN
X = np.concatenate((s1_binned.reshape((nt1,1)),s2_binned.reshape((nt1,1))))
y = np.concatenate((np.ones((nt1,)),2*np.ones((nt1,))))



beta, beta_0 = initialise(X_train.shape[0])

parameters, grads, costs = optimise(X_train, y_train, beta, beta_0, num_iterations, learning_rate)
    
beta = parameters["beta"]
beta_0 = parameters["beta_0"]
    
y_pred_test = predict_logistic(X_test, beta, beta_0, threshold)
