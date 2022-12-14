#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 10:06:20 2022

@author: ir318
"""

import numpy as np
import matplotlib.pyplot as plt
from defined_functions import *

def loglikelihood_linear(x,y,beta0,beta1,sigmasq):
    helper = beta0 + x * beta1
    twosigmasq = 2 * sigmasq
    log_lik = -len(x)/2 * np.log(twosigmasq*np.pi) - sum((y-helper)**2) / twosigmasq
    return log_lik

beta_0_true = 0.4
beta_1_true = 0.6
size = 500

mean = 5
variance = 3

how_many = 1000
bin_size = 1
bins = np.arange(bin_size/2,how_many,bin_size)
bins = np.concatenate((-bins[::-1], bins))

list_beta0 = np.linspace(0,1,200)
np.savetxt("list_beta0.csv", list_beta0, delimiter=",")   

list_beta1 = np.linspace(0.5,0.9,900)
np.savetxt("list_beta1.csv", list_beta1, delimiter=",")

#### FIRST METHOD, WHEN I ASSIGN SIGMA^2 = VARIANCE STRAIGHTAWAY

# matrix_log_lik = np.zeros((len(list_beta0),len(list_beta1)))
# matrix_log_lik_binned = np.zeros((len(list_beta0),len(list_beta1)))

# for index1 in range(len(list_beta0)):
#     for index2 in range(len(list_beta1)):
#         matrix_log_lik[index1,index2] = loglikelihood_linear(X,y,list_beta0[index1],list_beta1[index2],variance)
#         matrix_log_lik_binned[index1,index2] = loglikelihood_linear(new_X,y,list_beta0[index1],list_beta1[index2],variance)

# np.savetxt("matrix_log_lik.csv", matrix_log_lik, delimiter=",")
# np.savetxt("matrix_log_lik_binned.csv", matrix_log_lik_binned, delimiter=",")

# xaxis, yaxis = np.meshgrid(list_beta0, list_beta1)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.contour3D(xaxis, yaxis, matrix_log_lik, 1000, cmap = 'Blues')
# ax.contour3D(xaxis, yaxis, matrix_log_lik_binned, 1000, cmap = 'Reds')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z');
# plt.show()

#### SECOND METHOD, WHEN I ASSIGN SIGMA^2 = 1/n sum((yi-beta0-beta1 *nx)**2)

matrix_log_lik2 = np.zeros((len(list_beta0),len(list_beta1)))
matrix_log_lik_binned2 = np.zeros((len(list_beta0),len(list_beta1)))
number_iterations = 1

for iterations in range(number_iterations):
    X = np.random.normal(mean, variance, size)
    y = beta_0_true + beta_1_true * X + np.random.normal(0, 1, size)
    new_X = put_in_bins(X, bins) 
    correction = (1 - 1/12 * 1/ np.var(new_X) **2 ) ** (-1)
    print('correction is',correction)
    for index1 in range(len(list_beta0)):
        for index2 in range(len(list_beta1)):
            sigmasq = 1 / size * sum( (y - list_beta0[index1] - X * list_beta1[index2] )**2)
            matrix_log_lik2[index1,index2] += loglikelihood_linear(X,y,list_beta0[index1],list_beta1[index2],sigmasq)
            sigmasq_binned = 1 / size * sum( (y - list_beta0[index1] - new_X * list_beta1[index2] )**2)
            matrix_log_lik_binned2[index1,index2] += loglikelihood_linear(new_X,y,list_beta0[index1],list_beta1[index2],sigmasq_binned)

matrix_log_lik2/= number_iterations
matrix_log_lik_binned2/= number_iterations

np.savetxt("matrix_log_lik2.csv", matrix_log_lik2, delimiter=",")
np.savetxt("matrix_log_lik_binned2.csv", matrix_log_lik_binned2, delimiter=",")

xaxis, yaxis = np.meshgrid(list_beta1, list_beta0)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xaxis, yaxis, matrix_log_lik2, 1000, cmap = 'Blues')
ax.contour3D(xaxis, yaxis, matrix_log_lik_binned2, 1000, cmap = 'Reds')
ax.set_xlabel('$\\beta_1$')
ax.set_ylabel('$\\beta_0$')
ax.set_zlabel('$l(\\beta_0, \\beta_1, \\sigma^2)$');
ax.zaxis.labelpad=10
ax.azim = -60
ax.dist = 10
ax.elev = 20
plt.title('Plot of Log-likelihood Surface of Linear Regression Model')
proxy = [plt.Rectangle((1, 1), 2, 2, fc=pc) for pc in ['blue','red']]
plt.legend(proxy, ["unbinned", "binned"])
plt.show()

indeces_max_bounded = np.where(matrix_log_lik_binned2== matrix_log_lik_binned2.max() )
indeces_max = np.where(matrix_log_lik2 == matrix_log_lik2.max() )
beta0hat = list_beta0[indeces_max[0][0]]
beta0hat_binned = list_beta0[indeces_max_bounded[0][0]]
beta1hat = list_beta1[indeces_max[1][0]]
beta1hat_binned = list_beta1[indeces_max_bounded[1][0]]

corrected_gradient = correction * beta1hat_binned

xaxis, yaxis = np.meshgrid(list_beta1, list_beta0)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.contour3D(xaxis, yaxis, matrix_log_lik2 -matrix_log_lik_binned2 , 1000, cmap = 'Blues')
ax.set_xlabel('$\\beta_1$')
ax.set_ylabel('$\\beta_0$')
ax.set_zlabel('$l_u - l_b$');
ax.zaxis.labelpad=10
ax.azim = -60
ax.dist = 10
ax.elev = 20
plt.title('Plot of Difference in Log-likelihood Surfaces using \n Unbinned vs Binned Data')
plt.show()
