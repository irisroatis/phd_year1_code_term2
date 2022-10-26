#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 09:57:37 2022

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

mu1, sigma1 = [0], [1] # mean and standard deviation
mu2, sigma2 = [2], [1] # mean and standard deviation

def put_in_bins(data, bins):
    digitized = np.digitize(data,bins)
    midpoints_bins = (bins[:len(bins)-1] + bins[1:])/2
    new_data = midpoints_bins[digitized-1]
    return new_data

iterations = 5000
how_many_times_repeat = 20

list_nt = [5, 15, 40, 85, 200, 500]
list_b = [2, 1, 0.5,0.35, 0.2, 0.1, 0.075, 0.05, 0.0001]

# keeping track on which iteration we are on
total_number_of_cases = len(list_nt)*len(list_b)
progress = 0

store_proportions_lda = np.zeros((len(list_nt),len(list_b)))
store_proportions_knn = np.zeros((len(list_nt),len(list_b)))



# simulate the data for testing

testing_data=[]
belonging_classes=[]

for repeat in range(how_many_times_repeat):

    random_simulation = np.zeros((iterations,))
    which_class_list = np.zeros((iterations,))
    
    for itera in range(iterations):

        which_normal = random.randint(1,2)
        if which_normal == 1:
            random_simulation[itera,] = np.random.normal(mu1, sigma1, (1))
        else:
            random_simulation[itera,] = np.random.normal(mu2, sigma2, (1))
        which_class_list[itera,] = which_normal
    
    testing_data.append(random_simulation)
    belonging_classes.append(which_class_list)

print('Testing data generated.')


for nt_index in range(len(list_nt)):
    
    nt = list_nt[nt_index]
  
    size1 = nt
    size2 = nt

    # generate from normal distrib
    s1 = np.random.normal(mu1, sigma1, size1)
    s2 = np.random.normal(mu2, sigma2, size2)


    for bin_size_index in range(len(list_b)):

        bin_size = list_b[bin_size_index]
        
        bins = np.arange(-100,100,bin_size)


        s1_binned = put_in_bins(s1, bins)
        s2_binned = put_in_bins(s2, bins)


        # create X and y to apply LDA
        X = np.concatenate((s1_binned.reshape((size1,1)),s2_binned.reshape((size2,1))))
        y = np.concatenate((np.ones((size1,)),2*np.ones((size2,))))
        model_lda = LinearDiscriminantAnalysis()
        model_lda.fit(X, y)
        
        model_knn = KNeighborsClassifier(n_neighbors=5)
        model_knn.fit(X, y)



        for repeat in range(how_many_times_repeat):
            
            testing_model_on = put_in_bins(testing_data[repeat],bins)
            correct_classes = belonging_classes[repeat]
            
            predicted_classes_lda  = model_lda.predict(testing_model_on.reshape(-1,1))
            predicted_classes_knn  = model_knn.predict(testing_model_on.reshape(-1,1))
            
            store_proportions_lda[nt_index,bin_size_index] +=  sum(predicted_classes_lda == correct_classes) / iterations
            store_proportions_knn[nt_index,bin_size_index] +=  sum(predicted_classes_knn == correct_classes) / iterations

        
        progress +=1
        print(progress/total_number_of_cases)
        
store_proportions_lda /= how_many_times_repeat
store_proportions_knn /= how_many_times_repeat







fig = plt.figure()
ax = plt.axes(projection='3d')
x,y = np.meshgrid(np.array(list_b),np.array(list_nt))
p=ax.scatter(x, y, store_proportions_lda, c=store_proportions_lda, cmap='viridis', linewidth=0.5);
ax.set_xlabel('size train samples')
ax.set_ylabel('bin size')
ax.set_zlabel('accuracy')
ax.set_title('LDA')
fig.colorbar(p,ax=ax,anchor=(1.0,0.0))
plt.show()

plt.figure()
plt.scatter(x,y,c=store_proportions_lda)
plt.xlabel('bin size')
plt.ylabel('size train samples')
plt.colorbar()
plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
x,y = np.meshgrid(np.array(list_b),np.array(list_nt))
p=ax.scatter(x, y, store_proportions_knn, c=store_proportions_knn, cmap='viridis', linewidth=0.5);
ax.set_xlabel('size train samples')
ax.set_ylabel('bin size')
ax.set_zlabel('accuracy')
ax.set_title('kNN')
fig.colorbar(p,ax=ax,anchor=(1.0,0.0))
plt.show()

plt.figure()
plt.scatter(x,y,c=store_proportions_knn)
plt.xlabel('bin size')
plt.ylabel('size train samples')
plt.colorbar()
plt.show()

