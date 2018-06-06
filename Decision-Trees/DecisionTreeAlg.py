# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 17:15:58 2018

@author: Sateesh
"""

""" DECISION TREE ALGO """

# Problem Description:
"""  """

#%%
''' Gini Index = sum(weight(1 - proportion^2)) . Sum is over the split datasets. Weight is the size of the split dataset in relation to the parent dataset.'''
def gini_index(left_group, right_group):
    size = left_group.count(axis = 0)[0] + right_group.count(axis = 0)[0] # Size of the dataset at the parent node
    gini = 0
    for group in left_group, right_group:
        labels = list(set(group['class']))
        rows = group.count(axis = 0)[0]     # Size of the split dataset
        
        sumProp = 0                         # Sum of squares of proportions of each class
        for label in labels:
            proportion = group['class'].value_counts()[label]/rows
            sumProp = sumProp + proportion*proportion
        
        gini = gini + (rows/size)*(1-sumProp)   
    return gini

def make_split(attribute, value, group):
    left_group = group[group[attribute] < value]
    right_group = group[group[attribute] >= value]
    return left_group, right_group

def split_data(group):
    gini = []                          # Gini score tracker
    attributes = group.columns[:-1]
    for attribute in attributes:
        values = list(group[attribute].unique())
        for value in values:
            left_group, right_group = make_split(attribute, value, group)  # make_split
            gini.append(gini_index(left_group, right_group))               # gini_score
            print(attribute, value, gini[-1])
            if gini[-1] == min(gini):
                node_attrib, node_value = attribute, value
                best_left_group = left_group 
                best_right_group = right_group
    return {'attrib': node_attrib, 'value': node_value, 'left_group' : best_left_group, 'right_group': best_right_group}

def to_terminal(group):
    outcome = group['class'].value_counts().index[0]
    return outcome

def split(node, max_depth, min_num_samples, depth):
    left, right = node['left_group'], node['right_group']
    del(node['left_group'])
    del(node['right_group'])
    
    if left.count(axis = 0)[0] == 0 or right.count(axis = 0)[0] == 0:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if left.count(axis = 0)[0] <= min_num_samples:
        node['left'] = to_terminal(left)
    else:
        node['left'] = split_data(left)
        split(node['left'], max_depth, min_num_samples, depth+1)
    if right.count(axis = 0)[0] <= min_num_samples:
        node['right'] = to_terminal(left)
    else:
        node['right'] = split_data(right)
        split(node['right'], max_depth, min_num_samples, depth+1)

def build_tree(data, max_depth, min_num_samples):
    root = split_data(data)
    split(root, max_depth, min_num_samples, 1)
    return root

def predict(node, sample):
    #print(count,'.)',node,'\n')
    if sample[node['attrib']] < node['value']:
        if isinstance(node['left'], dict):
            prediction = predict(node['left'], sample)
        else:
            #print(node['left'],'\n') 
            return node['left']
  
    else:
        if isinstance(node['right'], dict):
            prediction = predict(node['right'], sample)
        else:
            #print(node['right'],'\n')
            return node['right']
    print('I come here\n')
    return prediction

def fit_decision_tree(train_data, max_depth, min_num_samples):
    tree = build_tree(train_data, max_depth, 1)
    return tree

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

def import_dependencies():
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
#%%
if __name__ == '__main__' :
    import_dependencies()
    data = pd.read_csv('data_banknote_authentication.csv', header = None, names = ['Variance', 'Skewness', 'curtosis', 'entropy', 'class'])
    # Need to divide data into training and testing splits
    feature_data, target = data[data.columns[:-1]], data[data.columns[-1]]
    
    train_X, test_X, train_y, test_y = train_test_split(feature_data, target)
    
    train_data = train_X
    train_data['class'] = train_y
    
    accur = []
    for max_depth in range(1,4):
        tree = fit_decision_tree(train_data, max_depth, 1)
        predictions = decision_tree_predictions(tree, test_X)
        accur.append(accuracy(predictions, test_y))
        
    iter = 0    
    for max_depth in range(1,4):
        print('max_depth:',max_depth, 'min_samples:',1, 'accuracy:',accur[iter])
        iter = iter + 1