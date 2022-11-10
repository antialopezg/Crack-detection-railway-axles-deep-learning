# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 12:38:20 2022

@author: antia
"""
import pandas as pd
import numpy as np

# DEFINE PATHS TO DATA 

path = './results_ts/'
path_ws1 = 'data/tsfresh_data/f1.pkl'
path_ws2 = 'data/tsfresh_data/f2.pkl'
path_ws3 = 'data/tsfresh_data/f3.pkl'
path_ws4 = 'data/tsfresh_data/f4.pkl'

print("Reading data")
ws1 = pd.read_pickle(path_ws1)
ws2 = pd.read_pickle(path_ws2)
ws3 = pd.read_pickle(path_ws3)
ws4 = pd.read_pickle(path_ws4)

ws1 = ws1.dropna()
ws2 = ws2.dropna()
ws3 = ws3.dropna()
ws4 = ws4.dropna()
#%%

def preprocess_pickle_configuration(dataset, lado, direction, corte, carga, velocidad):
    data = dataset
    data['Load'] = data['Load'].round()

    data = data[(data['Lado']==lado) & (data['Direction']==direction) & (data['Load']==carga) 
                & (data['Corte']==corte) & (data['Velocidad']==velocidad)]
    
    feat = data[['Lado','Direction','Corte','Load','Velocidad']]
    Y = data['Label']  
    #nofeat = ['Lado','Direction','Corte','Load','Velocidad', 'Label']
    nofeat= ['Label']
    X = data.drop(columns=nofeat)
    
    return X, Y, feat

#%% TRAIN VALID TEST SPLITTING 
from sklearn.model_selection import train_test_split, ParameterGrid
print("Splitting data")

ws1_train, ws1_test = train_test_split(ws1, test_size=0.3, random_state=42)
y_tr = ws1_train['Label']
x_tr = ws1_train.drop(columns=['Label'])
y_test = ws1_test['Label']
x_test = ws1_test.drop(columns=['Label'])

#%% RANDOM FOREST 

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
print("Training gridSearchCV for k=5")
grid_search.fit(x_tr, y_tr)
print("Scoring results")
acc = grid_search.score(x_test, y_test)
auc = roc_auc_score(y_test, grid_search.predict_proba(x_test), multi_class='ovr')
print('Acc for test WS1', acc)
print('Auc for test WS1', auc)

#%%

# DEFINE ALL POSSIBLE CONFIGURATIONS 
configurations = {'Load': [0.0,1.0],'Velocidad': [0.0,1.0],'Lado' : [0,1],'Direction':[0,1],'Corte': [0,1]}

grid = list(ParameterGrid(configurations))

# TABLE TO SAVE RESULTS
results_rf = pd.DataFrame(columns=['Lado','Corte','Direction','Load','Velocidad',
                                'ACC WS1', 'ACC WS2', 'ACC WS3', 'ACC WS4',
                                'AUC WS1', 'AUC WS2', 'AUC WS3', 'AUC WS4'])

print("Testing all configurations")
#  ITERATE AND SAVE RESULTS
for conf in grid:
    
    # OBTAIN ALL DATA FOR EACH SUBSET
    X_ws1, Y_ws1, feat_ws1 = preprocess_pickle_configuration(ws1_test, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
    X_ws2, Y_ws2, feat_ws2 = preprocess_pickle_configuration(ws2, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
    X_ws3, Y_ws3, feat_ws3 = preprocess_pickle_configuration(ws3, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
    X_ws4, Y_ws4, feat_ws4 = preprocess_pickle_configuration(ws4, lado=conf['Lado'], direction = conf['Direction'], corte=conf['Corte'], carga=conf['Load'],velocidad=conf['Velocidad'])
    

    # WS1 
    acc_ws1 = grid_search.score(X_ws1, Y_ws1)
    auc_ws1 = roc_auc_score(Y_ws1, grid_search.predict_proba(X_ws1), multi_class='ovr')

    # WS2 
    acc_ws2 = grid_search.score(X_ws2, Y_ws2)
    auc_ws2 = roc_auc_score(Y_ws2, grid_search.predict_proba(X_ws2), multi_class='ovr')
    
    # WS3
    acc_ws3 = grid_search.score(X_ws3, Y_ws3)
    auc_ws3 = roc_auc_score(Y_ws3, grid_search.predict_proba(X_ws3), multi_class='ovr')
    
    # WS4
    acc_ws4 = grid_search.score(X_ws4, Y_ws4)
    auc_ws4 = roc_auc_score(Y_ws4, grid_search.predict_proba(X_ws4), multi_class='ovr')
    
    # UPDATE TABLE AFTER EACH ITERATION FOR A SPECIFIC CONFIGURATION
    
    results_rf = results_rf.append({'Lado': conf['Lado'],
                            'Corte': conf['Corte'],
                            'Direction': conf['Direction'],
                            'Load': conf['Load'],
                            'Velocidad': conf['Velocidad'],
                            'ACC WS1': round(acc_ws1.item(),3),
                            'ACC WS2': round(acc_ws2.item(),3),
                            'ACC WS3': round(acc_ws3.item(),3),
                            'ACC WS4': round(acc_ws4.item(),3),
                            'AUC WS1': round(auc_ws1,3),
                            'AUC WS2': round(auc_ws2,3),
                            'AUC WS3': round(auc_ws3,3),
                            'AUC WS4': round(auc_ws4,3)}, ignore_index=True)

results_rf.to_csv(path + 'results_rf.csv')
