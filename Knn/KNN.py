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
    data = pd.read_csv('iris.csv', header = None)
    return data 

def get_splits(data):
    train_data, test_data = train_test_split(data, random_state = 42)
    validation_data = test_data.copy()
    validation_class = validation_data.loc[:,4]
    validation_data.drop(4, axis = 1, inplace = True)
    return train_data, validation_data, validation_class

def eucledian_distance(instance, training_point):
    squaredDistance = np.square(instance - training_point).sum()
    return np.sqrt(squaredDistance)

def get_nearest_neighbors(instance, train_data, k):
    distance, referenceIndex = [], []
    for i in range(len(train_data)):
        distance.append(eucledian_distance(instance, train_data.iloc[i][train_data.columns[:-1]]))
        referenceIndex.append(train_data.index[i])
    distanceFrame = pd.DataFrame({'ReferenceIndex': referenceIndex, 'Distance':distance})       
    distanceFrame.sort_values(by = 'Distance', axis = 0, kind = 'quicksort', ascending = True, inplace = True)
    nearestNeighborsIndex = distanceFrame['ReferenceIndex'][:k]
    nearestNeighbors = train_data.loc[nearestNeighborsIndex]
    #print(distanceFrame)
    #print(nearestNeighbors)
    return nearestNeighbors
    
def get_predictions(sample, train_data, k):
    nearestNeighbors = get_nearest_neighbors(sample, train_data, k)
    prediction = nearestNeighbors[4].value_counts().index[0]
    return prediction

def get_accuracy(predictions, validation_class):
    validation_class = validation_class.reset_index()
    correct = 0
    #print(predictions)
    for i in range(len(predictions)):
        if predictions[i] == validation_class.iloc[i][4]:
            correct+=1
    return 100*correct/len(predictions)
    
def optimal_k(train_data, validation_data, validation_class):
    scores = []
    for k in range(2,5):
        predictions = []
        for sample_index in range(len(validation_data)):
            sample = validation_data.iloc[sample_index]
            #print(sample)
            predictions.append(get_predictions(sample, train_data, k))
        #print(predictions)
        scores.append(get_accuracy(predictions, validation_class))
        print(k,':',scores[-1])
    k_req = scores.index(max(scores)) + 7
    print(scores)
    return k_req

def knn_prediction(train_data, test_data, k):
    predictions = []
    for sample_index in range(len(validation_data)):
        sample = validation_data.iloc[sample_index]
        predictions.append(get_predictions(sample, train_data, k))
    return predictions
    
#%%
if __name__ == '__main__':
    data = get_data()
    train_data, validation_data, validation_class = get_splits(data)
    k_req = optimal_k(train_data, validation_data, validation_class)
    predictions = knn_prediction(train_data, validation_data, k_req)
    accur = get_accuracy(predictions, validation_class)    