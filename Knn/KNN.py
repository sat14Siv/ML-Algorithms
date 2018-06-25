# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:24:38 2018

@author: Sateesh
"""

"""K- Nearest Neighbors Algorithm """
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#%% Functions
def get_data():
    data = pd.read_csv('pima.csv')
    return data 

def get_splits(data):     # Dividing the data into training and validation splits
    train_data, test_data = train_test_split(data, random_state = 42)
    target_column = train_data.columns[-1]
    validation_data = test_data.copy()
    validation_class = validation_data.loc[:,target_column]
    validation_data.drop(target_column, axis = 1, inplace = True)
    return train_data, validation_data, validation_class

def euclidian_distance(instance, training_point):  # Using Euclidean to measure distance between points 
    squaredDistance = np.square(instance - training_point).sum()
    return np.sqrt(squaredDistance)

def get_nearest_neighbors(instance, train_data, k): # Returns the nearest neighbors of an instance
    distance, referenceIndex = [], []
    for i in range(len(train_data)):
        distance.append(euclidian_distance(instance, train_data.iloc[i][train_data.columns[:-1]]))
        referenceIndex.append(train_data.index[i])
    distanceFrame = pd.DataFrame({'ReferenceIndex': referenceIndex, 'Distance':distance})       
    distanceFrame.sort_values(by = 'Distance', axis = 0, kind = 'quicksort', ascending = True, inplace = True)
    nearestNeighborsIndex = distanceFrame['ReferenceIndex'][:k]
    nearestNeighbors = train_data.loc[nearestNeighborsIndex]
    return nearestNeighbors
    
def get_predictions(sample, train_data, k):   # Returns the predicted target/ class of an instance
    target_column = train_data.columns[-1]
    nearestNeighbors = get_nearest_neighbors(sample, train_data, k)
    prediction = nearestNeighbors[target_column].value_counts().index[0] # Most ocurring class in the neighbors is returned
    print(sample)
    return prediction

def get_accuracy(predictions, validation_class):
    validation_class = validation_class.reset_index()
    target_column =  validation_class.columns[-1]
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == validation_class.iloc[i][target_column]:
            correct+=1
    return 100*correct/len(predictions)
    
def optimal_k(train_data, validation_data, validation_class, k_min, k_max): 
# Function to find out the optimal k to be used. Evaluates the validation accuracy for each k and returns the k with maximum accuracy.`
    scores = []
    for k in range(k_min, k_max+1):
        predictions = []
        for sample_index in range(len(validation_data)):
            sample = validation_data.iloc[sample_index]
            predictions.append(get_predictions(sample, train_data, k))
        scores.append(get_accuracy(predictions, validation_class))
        print(k,':',scores[-1])
    k_req = scores.index(max(scores)) + k_min
    return k_req

def knn_prediction(train_data, test_data, k): # After finding out the optimal k for the data, this is use to classify the test data.
    predictions = []
    for sample_index in range(len(validation_data)):
        sample = validation_data.iloc[sample_index]
        predictions.append(get_predictions(sample, train_data, k))
    return predictions 
#%%
if __name__ == '__main__':
    data = get_data()
    train_data, validation_data, validation_class = get_splits(data)
    #k_req = optimal_k(train_data, validation_data, validation_class, 2, 5)
    # Now, use the test data. In my case, there is no test data
    predictions = knn_prediction(train_data, validation_data, 4)
    accur = get_accuracy(predictions, validation_class)  
    print(accur)