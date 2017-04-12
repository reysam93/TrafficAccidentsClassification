# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:26:57 2017

@author: samuel
"""

import pandas as pd

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Read and shuffle data
train_data = shuffle(pd.read_csv('traindata.csv'), random_state=0)
print("Data:", train_data.shape)

# If there are any dead or injured -> class 1
Y = []
Y_labels = ["N_Muertos", "N_Graves", "N_Leves"]
cont = 0
for val in train_data[Y_labels].values:
    if val.any() > 0:
        Y.append(1)
        cont += 1
    else:
        Y.append (0)

# Checking if the class are balanced
print("P Class 1:", cont/float(len(Y)))

X = train_data.drop(Y_labels, axis=1)

# Remove colums with no interest and srting values and nan values
rm_cols = ["Id", "Numero_Accidente", "Carretera", "Pk", "Km", "Tipo_Est"]
X.drop(rm_cols, axis=1, inplace=True)
X.fillna(value=0, inplace=True)

# Split in train and validation
# Maybe shuffle is not neccesary if using random_state here
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)

# Normalization
scalar = StandardScaler()
scalar.fit(X_train)
X_train_n = scalar.transform(X_train)
X_val_n = scalar.transform(X_val)

# MLP creation + trainning
mlp = MLPClassifier(activation="logistic", learning_rate="adaptive", verbose=True,
                    max_iter=300, hidden_layer_sizes=30)
mlp.fit(X_train_n, Y_train)

# Prediction and evaluation
prediction = mlp.predict(X_val_n)
print(classification_report(Y_val, prediction))
print(confusion_matrix(Y_val, prediction))