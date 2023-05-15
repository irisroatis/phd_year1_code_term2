#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 20:51:38 2023

@author: roatisiris
"""


import pandas as pd
from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn import preprocessing
import random
import numpy as np


def dataset_to_Xandy(dataset, target_variable):
    X = dataset.loc[:, dataset.columns != target_variable]
    y = dataset.loc[:, dataset.columns == target_variable]
    return X, y

def split_dataset(X,y, randomlist, not_in_randomlist):
    X_train = X.iloc[randomlist,:]
    y_train = y.iloc[randomlist,:]
    X_test = X.iloc[not_in_randomlist,:]
    y_test = y.iloc[not_in_randomlist,:]
    return X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy()

def plot_conf_matrix(confusion_matrix,encoding):
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
    cm_display.plot()
    plt.title('Confusion matrix for '+str(encoding)+' encoding')
    plt.show()  

def calc_conf_matrix(X,y,randomlist,not_in_randomlist):
    X_train, y_train, X_test, y_test = split_dataset(X,y,randomlist, not_in_randomlist)
    logistic_regressor = LogisticRegression(penalty = 'none')  # create object for the class
    logistic_regressor.fit(X_train, y_train.reshape(-1,))  # perform linear regression
    y_predicted = logistic_regressor.predict(X_test)
    matrix = metrics.confusion_matrix(y_test, y_predicted)
    fpr, tpr, _ = metrics.roc_curve(y_test, y_predicted)
    area_roc = metrics.auc(fpr, tpr)
    return matrix[0,0], matrix[0,1], matrix[1,0], matrix[1,1], area_roc

def plot_boxplots_confusion(confusion_matrix,entry):
    dictionary = {}
    for key in confusion_matrix:
        dictionary[key] = confusion_matrix[key][entry]
    fig, ax = plt.subplots()
    ax.boxplot(dictionary.values())
    ax.set_xticklabels(dictionary.keys())
    if entry == '00':
        name = 'true negative'
    elif entry == '01':
        name = 'false positive'
    elif entry == '10':
        name = 'false negative'
    elif entry == '11':
        name = 'true positive'
    elif entry == 'accuracy':
        name = 'accuracy'
    elif entry == 'auc':
        name = 'area under ROC curve'
    plt.title('Boxplots of ' + name +' \n Test Size '+ str(out_of))
    plt.show()    


######### HEART DATASET    



# df = pd.read_csv('datasets/heartdataset.csv')
# categorical_cols = ['cp','thal','slope','ca','restecg'] # Putting in this all the categorical columns
# target_variable = 'target' # Making sure the name of the target variable is known




######### AIRLINE DATASET  

df = pd.read_csv('datasets/airline_dataset.csv')
categorical_cols = ['MONTH','DAY_OF_WEEK','DEP_TIME_BLK','DISTANCE_GROUP','SEGMENT_NUMBER','CARRIER_NAME', 'DEPARTING_AIRPORT','PREVIOUS_AIRPORT'] # Putting in this all the categorical columns
target_variable = 'DEP_DEL15' # Making sure the name of the target variable is known

df0 = df.loc[df[target_variable] ==0 ]
df1 = df.loc[df[target_variable] ==1 ]
how_many_0 = df0.shape[0]
how_many_1 = df1.shape[0]
random_indices = random.sample(range(0, how_many_0), how_many_0 - 4000)
df0 = df0.drop(df0.index[random_indices])
random_indices = random.sample(range(0, how_many_1), how_many_1 - 4000)
df1 = df1.drop(df1.index[random_indices])
df = pd.concat([df0, df1])






######### CAR INSURANCE DATASET  

# df = pd.read_csv('datasets/cardataset.csv')
# df = df.drop('policy_id',axis = 1)
# categorical_cols = ['area_cluster','make', 'segment','model', 'fuel_type','max_torque','max_power','engine_type','steering_type','ncap_rating'] # Putting in this all the categorical columns
# target_variable = 'is_claim' # Making sure the name of the target variable is known

# binary_cols = ['gear_box','is_esc','is_adjustable_steering','is_tpms',
#                 'is_parking_sensors','is_parking_camera','rear_brakes_type',
#                 'cylinder','transmission_type','is_front_fog_lights'
#                 ,'is_rear_window_wiper','is_rear_window_washer'
#                 ,'is_rear_window_defogger', 'is_brake_assist', 'is_power_door_locks',
#                 'is_central_locking','is_power_steering','is_driver_seat_height_adjustable',
#                 'is_day_night_rear_view_mirror','is_ecw','is_speed_alert']

# ### make sure binary variables are 0 and 1
# labelencoder = ce.OrdinalEncoder(cols=binary_cols)
# df = labelencoder.fit_transform(df)





########### WINE QUALITY


# df = pd.read_csv('datasets/wine_dataset.csv')
# target_variable = 'quality' # Making sure the name of the target variable is known
# df[target_variable] = df[target_variable].replace(['bad'], 0)
# df[target_variable] = df[target_variable].replace(['good'], 1)

# categorical_cols = ['free sulfur dioxide', 'pH', 'alcohol', 'citric acid'] # Putting in this all the categorical columns
# for keys in df:
#     print(df[keys].value_counts())



# ########### STROKE PREDICTION
# df = pd.read_csv('datasets/stroke_dataset.csv')
# df = df.drop('id',axis = 1)
# df = df.dropna().reset_index(drop=True)
# df = df.replace('*82', 82)
# categorical_cols = ['gender','work_type','smoking_status'] # Putting in this all the categorical columns
# target_variable = 'stroke' # Making sure the name of the target variable is known

# binary_cols = ['Residence_type','ever_married','heart_disease','hypertension']

# ### make sure binary variables are 0 and 1
# labelencoder = ce.OrdinalEncoder(cols=binary_cols)
# df = labelencoder.fit_transform(df)




# ########### BODY SIGNAL SMOKING
# df = pd.read_csv('datasets/bodysignal_smoking.csv')
# df = df.drop(['ID','oral'],axis = 1)


# categorical_cols = ['Urine protein','eyesight(right)','eyesight(left)'] # Putting in this all the categorical columns
# target_variable = 'smoking' # Making sure the name of the target variable is known

# binary_cols = ['tartar','dental caries','hearing(right)','hearing(left)','gender']

# ### make sure binary variables are 0 and 1
# labelencoder = ce.OrdinalEncoder(cols=binary_cols)
# df = labelencoder.fit_transform(df)






###### START 

size = df.shape[0] # size of the dataset
# Seeing if they are indeed categorical
for cat in categorical_cols:
    print(df[cat].value_counts())

# Seeing if the data is balanced
plt.figure()
df[target_variable].value_counts().plot(kind='bar')
plt.title('Test how many target 0 vs 1')
plt.show()


how_many_iterations = 20 # how many CV folds

# initialising confusion matrices

confusion_matrix = {}
confusion_matrix['no_cat'] =  {'00':np.zeros((how_many_iterations,)),'10':np.zeros((how_many_iterations,)),'01':np.zeros((how_many_iterations,)),'11':np.zeros((how_many_iterations,)),
                               'accuracy':np.zeros((how_many_iterations,)), 'auc':np.zeros((how_many_iterations,))}
confusion_matrix['simple'] =   {'00':np.zeros((how_many_iterations,)),'10':np.zeros((how_many_iterations,)),'01':np.zeros((how_many_iterations,)),'11':np.zeros((how_many_iterations,)),
                               'accuracy':np.zeros((how_many_iterations,)), 'auc':np.zeros((how_many_iterations,))}
confusion_matrix['onehot'] =  {'00':np.zeros((how_many_iterations,)),'10':np.zeros((how_many_iterations,)),'01':np.zeros((how_many_iterations,)),'11':np.zeros((how_many_iterations,)),
                               'accuracy':np.zeros((how_many_iterations,)), 'auc':np.zeros((how_many_iterations,))}
confusion_matrix['target'] =   {'00':np.zeros((how_many_iterations,)),'10':np.zeros((how_many_iterations,)),'01':np.zeros((how_many_iterations,)),'11':np.zeros((how_many_iterations,)),
                               'accuracy':np.zeros((how_many_iterations,)), 'auc':np.zeros((how_many_iterations,))}
confusion_matrix['effect'] =  {'00':np.zeros((how_many_iterations,)),'10':np.zeros((how_many_iterations,)),'01':np.zeros((how_many_iterations,)),'11':np.zeros((how_many_iterations,)),
                               'accuracy':np.zeros((how_many_iterations,)), 'auc':np.zeros((how_many_iterations,))}

out_of = size - 4 * size // 5


for iteration in range(how_many_iterations):
    
    # Randomising the CV fold
    randomlist = random.sample(range(0, size),  4 * size// 5)
    not_in_randomlist = list(set(range(0,size)) - set(randomlist))
 
    
    
    ### PREDICTION WITHOUT THE CATEGORICAL ONES
    
    X, y =  dataset_to_Xandy(df.drop(categorical_cols, axis = 1), target_variable)
    m0, m1, m2, m3, auc = calc_conf_matrix(X,y,randomlist,not_in_randomlist)
    confusion_matrix['no_cat']['00'][iteration] = m0
    confusion_matrix['no_cat']['01'][iteration] = m1
    confusion_matrix['no_cat']['10'][iteration] = m2
    confusion_matrix['no_cat']['11'][iteration] = m3
    confusion_matrix['no_cat']['accuracy'][iteration] = (m0+m3) / out_of
    confusion_matrix['no_cat']['auc'][iteration] = auc
    
    
  
    ### SIMPLE ENCODING

    labelencoder = ce.OrdinalEncoder(cols=categorical_cols)
    new_df = labelencoder.fit_transform(df)
    X,y =  dataset_to_Xandy(new_df, target_variable)
    m0, m1, m2, m3, auc = calc_conf_matrix(X,y,randomlist,not_in_randomlist)
    confusion_matrix['simple']['00'][iteration] = m0
    confusion_matrix['simple']['01'][iteration] = m1
    confusion_matrix['simple']['10'][iteration] = m2
    confusion_matrix['simple']['11'][iteration] = m3
    confusion_matrix['simple']['accuracy'][iteration] = (m0+m3) / out_of
    confusion_matrix['simple']['auc'][iteration] = auc
  
    
    ### ONE HOT ENCODING
    encoder = ce.OneHotEncoder(cols=categorical_cols,use_cat_names=True)
    new_df = encoder.fit_transform(df)
    X,y =  dataset_to_Xandy(new_df, target_variable)
    m0, m1, m2, m3, auc = calc_conf_matrix(X,y,randomlist,not_in_randomlist)
    confusion_matrix['onehot']['00'][iteration] = m0
    confusion_matrix['onehot']['01'][iteration] = m1
    confusion_matrix['onehot']['10'][iteration] = m2
    confusion_matrix['onehot']['11'][iteration] = m3
    confusion_matrix['onehot']['accuracy'][iteration] = (m0+m3) / out_of
    confusion_matrix['onehot']['auc'][iteration] = auc
    
    
    
    #### EFFECT ENCODING
    encoder = ce.sum_coding.SumEncoder(cols=categorical_cols,verbose=False)
    new_df = encoder.fit_transform(df)
    X,y =  dataset_to_Xandy(new_df, target_variable)
    m0, m1, m2, m3, auc = calc_conf_matrix(X,y,randomlist,not_in_randomlist)
    confusion_matrix['effect']['00'][iteration] = m0
    confusion_matrix['effect']['01'][iteration] = m1
    confusion_matrix['effect']['10'][iteration] = m2
    confusion_matrix['effect']['11'][iteration] = m3
    confusion_matrix['effect']['accuracy'][iteration] = (m0+m3) / out_of
    confusion_matrix['effect']['auc'][iteration] = auc
    
    ##### TARGET ENCODING
    
    TE_encoder = ce.TargetEncoder(cols=categorical_cols)
    new_df = TE_encoder.fit_transform(df, df[target_variable])
    X,y =  dataset_to_Xandy(new_df, target_variable)
    m0, m1, m2, m3, auc = calc_conf_matrix(X,y,randomlist,not_in_randomlist)
    confusion_matrix['target']['00'][iteration] = m0
    confusion_matrix['target']['01'][iteration] = m1
    confusion_matrix['target']['10'][iteration] = m2
    confusion_matrix['target']['11'][iteration] = m3
    confusion_matrix['target']['accuracy'][iteration] = (m0+m3) / out_of
    confusion_matrix['target']['auc'][iteration] = auc

plot_boxplots_confusion(confusion_matrix, '00')
plot_boxplots_confusion(confusion_matrix, '01')
plot_boxplots_confusion(confusion_matrix, '10')
plot_boxplots_confusion(confusion_matrix, '11')
plot_boxplots_confusion(confusion_matrix, 'accuracy')
plot_boxplots_confusion(confusion_matrix, 'auc')

