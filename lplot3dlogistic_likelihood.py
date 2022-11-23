#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 09:33:25 2022

@author: ir318
"""
import numpy as np
import matplotlib.pyplot as plt
from defined_functions import *

def loglikelihood(x,y,beta0,beta1):
    helper = beta0 + x * beta1
    log_lik = sum(y * helper - np.log(1+np.exp(helper)))
    return log_lik

mu1, sigma1 = [0.25], [1] # mean and standard deviation
mu2, sigma2 = [1], [1] # mean and standard deviation
nt1 = 1000
bin_size1 = 1
bins = np.arange(bin_size1/2,100,bin_size1)
bins = np.concatenate((-bins[::-1], bins))

s1 = np.random.normal(mu1, sigma1, nt1)
s2 = np.random.normal(mu2, sigma2, nt1)

s1_binned = put_in_bins(s1, bins) #binned
s2_binned = put_in_bins(s2, bins) #binned

x = np.concatenate((s1.reshape((nt1,)),s2.reshape((nt1,))))
y = np.concatenate((0*np.ones((nt1,)),np.ones((nt1,))))

x_binned = np.concatenate((s1_binned.reshape((nt1,)),s2_binned.reshape((nt1,))))

list_beta0 = np.linspace(-10,10,100)
np.savetxt("list_beta0.csv", list_beta0, delimiter=",")   

list_beta1 = np.linspace(-10,20,100)
np.savetxt("list_beta1.csv", list_beta1, delimiter=",")

matrix_log_lik = np.zeros((len(list_beta0),len(list_beta1)))
matrix_log_lik_binned = np.zeros((len(list_beta0),len(list_beta1)))

for index1 in range(len(list_beta0)):
    for index2 in range(len(list_beta1)):
        matrix_log_lik[index1,index2] = loglikelihood(x,y,list_beta0[index1],list_beta1[index2])
        matrix_log_lik_binned[index1,index2] = loglikelihood(x_binned,y,list_beta0[index1],list_beta1[index2])

np.savetxt("matrix_log_lik.csv", matrix_log_lik, delimiter=",")
np.savetxt("matrix_log_lik_binned.csv", matrix_log_lik_binned, delimiter=",")

xaxis, yaxis = np.meshgrid(list_beta0, list_beta1)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xaxis, yaxis, matrix_log_lik, 500, cmap = 'Blues')
ax.contour3D(xaxis, yaxis, matrix_log_lik_binned, 500, cmap = 'Reds')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');

