# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:15:58 2018

@author: Sateesh
"""

""" DECISION TREE ALGO """

# Problem Description:
""" We need to classify a bank note as authentic or not depending on a number of measures (4) taken from a photograph. A decision tree has been implemented from scratch for this classification problem. All the data is numeric so the algorithm is written accordingly. """

#%% Functions
''' Gini Index = sum(weight(1 - proportion^2)) . Sum is over the split datasets. Weight is the size of the split dataset in relation to the parent dataset.'''
def gini_index(left_group, right_group):
    size = left_group.count(axis = 0)[0] + right_group.count(axis = 0)[0] # Size of the dataset at the 
    gini = 0                                                              # parent node
    for group in left_group, right_group:
        labels = list(set(group['class']))
        rows = group.count(axis = 0)[0]     # Size of the split dataset
        
        sumProp = 0                         # Sum of squares of proportions of each class
        for label in labels:
            proportion = group['class'].value_counts()[label]/rows
            sumProp = sumProp + proportion*proportion
        
        gini = gini + (rows/size)*(1-sumProp)   
    return gini

def make_split(attribute, value, group):         # Splitting a node into left and right nodes
    left_group = group[group[attribute] < value]
    right_group = group[group[attribute] >= value]
    return left_group, right_group

def split_data(group):         # Identifies the best split by going through all possible splits
    print('MAKE SPLIT')
    gini = []                          # Gini score tracker
    attributes = group.columns[:-1]
    for attribute in attributes:
        values = list(group[attribute].unique())
        for value in values:
            left_group, right_group = make_split(attribute, value, group)  # make_split
            gini.append(gini_index(left_group, right_group))               # gini_score
        
            if (gini[-1] == min(gini)):                     # keeping track of the minimum value
                node_attrib, node_value = attribute, value
                best_left_group = left_group 
                best_right_group = right_group
                gini_min = gini[-1]
                
                if (gini[-1] == 0):
                    print('Attribute:',node_attrib,'| Value:',node_value,'| Gini:',gini_min)
                    return {'attrib': node_attrib, 'value': node_value, 'left_group' : best_left_group, 'right_group': best_right_group, 'Gini':gini_min}
                    
    print('Attribute:',node_attrib,'Value:',node_value,'Gini:',gini_min)
        
    return {'attrib': node_attrib, 'value': node_value, 'left_group' : best_left_group, 'right_group': best_right_group, 'Gini':gini_min}

def to_terminal(group):                 # Terminal node. Returns majority class as output
    print('LEAF')
    print(group)
    if len(group)==0:
        print('ENCOUNTERED')
    outcome = group['class'].value_counts().index[0]
    print(outcome)
    return outcome

def split(node, max_depth, min_num_samples, depth):   # Creates child nodes by checking the criteria
    left, right = node['left_group'], node['right_group']
    gini = node['Gini']
    del(node['left_group'])
    del(node['right_group'])
    
    if left.count(axis = 0)[0] == 0 or right.count(axis = 0)[0] == 0:
        print('DEPTH:',depth)
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if (depth >= max_depth) or (gini == 0):
        print('DEPTH:',depth)
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if left.count(axis = 0)[0] <= min_num_samples:
        print('DEPTH:',depth)
        node['left'] = to_terminal(left)
    else:
        print('DEPTH:',depth)
        node['left'] = split_data(left)
        split(node['left'], max_depth, min_num_samples, depth+1)
    if right.count(axis = 0)[0] <= min_num_samples:
        print('DEPTH:',depth)
        node['right'] = to_terminal(right)
    else:
        print('DEPTH:',depth)
        node['right'] = split_data(right)
        split(node['right'], max_depth, min_num_samples, depth+1)

def build_tree(data, max_depth, min_num_samples): # Tree is initialised with the root node and split
    print('DEPTH:0')                                # is called (which is a recursive function)
    root = split_data(data)                       
    split(root, max_depth, min_num_samples, 1)
    return root

def predict(node, sample):                  
    if sample[node['attrib']] < node['value']:
        if isinstance(node['left'], dict):       # If a node is a leaf, then it will be a single value,
            prediction = predict(node['left'], sample) # else it will be a dictionary
        else:
            return node['left']
  
    else:
        if isinstance(node['right'], dict):
            prediction = predict(node['right'], sample)
        else:
            return node['right']
        
    return prediction

def decision_tree_predictions(tree, test_data):
    predictions = list()
    
    for sample_num in range(len(test_data)):
        sample = test_data.iloc[sample_num]
        predictions.append(predict(tree, sample))
        
    return predictions

def accuracy(predictions, test_y):
    inacc = abs(predictions - test_y).sum()
    perc_inacc = 100*inacc/len(test_y)
    return (100 - perc_inacc)
#%%
if __name__ == '__main__' :
    import pandas as pd
    from sklearn.model_selection import train_test_split
    # Read data into a dataframe
    data = pd.read_csv('data_banknote_authentication.csv', header = None, names = ['Variance', 'Skewness', 'curtosis', 'entropy', 'class'])
    # Need to divide data into training and testing splits
    feature_data, target = data[data.columns[:-1]], data[data.columns[-1]]
    train_X, test_X, train_y, test_y = train_test_split(feature_data, target)

    train_data = train_X.copy()            
    train_data.loc[:,'class'] = train_y  #Combining the training data and training targets
    # Fit the tree on training data and predict for test data
    min_samples = 2
    accur = []
    for max_depth in range(3,4):
        tree = build_tree(train_data, max_depth, min_samples)
        predictions = decision_tree_predictions(tree, test_X)
        accur.append(accuracy(predictions, test_y))
        
    iter = 0    
    for max_depth in range(3,4):
        print('max_depth:',max_depth, '| min_samples:',1, '| accuracy:',accur[iter])
        iter = iter + 1