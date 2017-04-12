# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 06:31:27 2017

@author: David
"""

"""
Load Data
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

X = pd.read_csv('traindata.csv')

cont = 0
for col in dataset["N_Muertos"]:
    if col > 0:
        cont += 1
        
X_test = pd.read_csv('testdata.csv')
 
print(cont)
print(len(X["N_Muertos"]))
print(X.describe())

#Separation in Train 80%, val 20%
X_train, X_val = np.split(X, [int(.8*len(X))])

"""
Normalice
"""

"""scaler = preprocessing.StandardScaler().fit(X)"""

"""
MPL
"""
"""
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10),solver='sgd',learning_rate_init=0.01,max_iter=500)

mlp.fit(X_train, Y_train)

print mlp.score(X_test,Y_test)   
"""                                     
