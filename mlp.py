# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:26:57 2017

@author: samuel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Needed to install neupy (sudo pip install neupy)
from sklearn import datasets, metrics
from neupy import algorithms, environment


# Not working!
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE



def print_all_data(X, Y):  
    np_y = np.array(Y)
    width = 0.35
    
    j = 1
    i = 0
    for col in X:
        values = X[col].values
        ind = set(values)
        ind2 = [x+width for x in ind]
        Y0 = []
        Y1 = []
        for val in ind:         
            ii = np.where(values == val)
            Y0.append(len(np.where(np_y[ii[0]] == 0)[0]))
            Y1.append(len(np.where(np_y[ii[0]] == 1)[0]))
            
        fig = plt.figure(i)
        ax = fig.add_subplot(2, 5, j)
        class0 = plt.bar(ind, Y0, width, color="y")  
        class1 = plt.bar(ind2, Y1, width, color="r")
        plt.xlabel(col)
        ax.set_xticklabels(ind)
        plt.legend((class0, class1), ("Class 0", "Class 1"))
        
        if j == 10:
            j = 1
            i += 1
        else:
            j += 1
    

def print_column(tag, X, Y):
    np_y = np.array(Y)
    width = 0.35
    
    values = X[tag].values
    ind = set(values)
    ind2 = [x+width for x in ind]
    Y0 = []
    Y1 = []
    for val in ind:         
        ii = np.where(values == val)
        Y0.append(len(np.where(np_y[ii[0]] == 0)[0]))
        Y1.append(len(np.where(np_y[ii[0]] == 1)[0]))
        
    fig = plt.figure(0)
    class0 = plt.bar(ind, Y0, width, color="y")  
    class1 = plt.bar(ind2, Y1, width, color="r")
    plt.xlabel(tag)
    plt.legend((class0, class1), ("Class 0", "Class 1"))


"""
Apparently interesting fields:
    
"""
def analyze_select_data(data):
    # If there are any dead or injured -> class 1
    Y = []
    Y_labels = ["N_Muertos", "N_Graves", "N_Leves"]
    for val in data[Y_labels].values:
        if val.any() > 0:
            Y.append(1)
        else:
            Y.append (0)
    
    # Checking if the class are balanced
    print("P Class 1 (Y):",  float(Y.count(1))/len(Y))
    print("P Class 0 (Y):",  float(Y.count(0))/len(Y))    
    
    X = data.drop(Y_labels, axis=1)
    
    # Remove colums with no interest and srting values and nan values
    rm_cols = ["Id", "Numero_Accidente", "Carretera", "Pk", "Km", "Tipo_Est",
                ]
    X.drop(rm_cols, axis=1, inplace=True)
    X.fillna(value=0, inplace=True)
    # print_all_data(X,Y) 
    print_column("Ano", X, Y)

    return X, Y


"""
    MLP network with the 80% if data for training and 20% for validation
"""
def mlpSimpleDiv(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)
    
    # Normalization
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train_n = scalar.transform(X_train)
    X_val_n = scalar.transform(X_val)
    
    # MLP creation + trainning
    mlp = MLPClassifier(activation="logistic", learning_rate="adaptive", verbose=False,
                        max_iter=150, hidden_layer_sizes=20, early_stopping=True, 
                        tol=1e-8, validation_fraction=0.3, alpha=1e-5)
                        
    # Prediction and evaluation
    mlp.fit(X_train_n, Y_train)
    prediction = mlp.predict(X_val_n)
    print(classification_report(Y_val, prediction, target_names=["Class 0", "Class 1"]))
    print(confusion_matrix(Y_val, prediction))
    return prediction, mlp


"""
    MLP network using cross validation for evaluating the model
"""
def mlpCrossVal(X, Y):
    
    #kbest = SelectKBest(f_classif, k=30)
    #X_red = kbest.fit_transform(X, Y)
    #vt = VarianceThreshold(0.9)
    #X_red = vt.fit_transform(X, Y)
    #print(X_red.shape)
    
    scalar = StandardScaler()
    mlp = MLPClassifier(activation="logistic", learning_rate="adaptive", verbose=True,
                        max_iter=200, hidden_layer_sizes=20, early_stopping=True, 
                        tol=1e-8, validation_fraction=0.3, alpha=1e-5)    
    pipeline = make_pipeline(scalar, mlp)
    scores = cross_val_score(pipeline, X, Y, cv=3)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores, mlp


def svmCrossVal(X, Y):
    #kbest = SelectKBest(f_classif, k=60)
    #X = kbest.fit_transform(X, Y)
    
    scalar = StandardScaler()
    svm = SVC(verbose=True, C=2)
    pipeline = make_pipeline(scalar, svm)
    scores = cross_val_score(pipeline, X, Y, cv=5)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores, svm


def kNeighborsCrossVal(X, Y):    
    scalar = StandardScaler()
    kn = KNeighborsClassifier(n_neighbors=200, p=2, weights="distance")
    pipeline = make_pipeline(scalar, kn)
    scores = cross_val_score(pipeline, X, Y, cv=5)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores, kn
    

def pnnTrainTestNoNorm(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
    pnn = algorithms.PNN(std=10, verbose=False)
    pnn.train(X_train, Y_train)
    
    y_predicted = pnn.predict(X_test)
    score = metrics.accuracy_score(Y_test, y_predicted)
    print("PNN score: ", score)
    return score, y_predicted
    
    
def pnnTrainTestNorm(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7)
    
    scalar = StandardScaler()
    scalar.fit(X_train)
    X_train_n = scalar.transform(X_train)
    X_test_n = scalar.transform(X_test)
    
    pnn = algorithms.PNN(std=10, verbose=False)
    pnn.train(X_train_n, Y_train)
    
    y_predicted = pnn.predict(X_test_n)
    score = metrics.accuracy_score(Y_test, y_predicted)
    print("PNN score: ", score)
    return score, y_predicted


"""
    There are several commented functions.
    Each of them makes the classification using different algorithms
    Uncomment the desired one
"""
if __name__ == "__main__":
    # Read and shuffle data
    train_data = shuffle(pd.read_csv('traindata.csv'), random_state=0)
    print("Data:", train_data.shape)
    
    X, Y = analyze_select_data(train_data)
    print("Shape of X:", X.shape)
    
    #mlpSimpleDiv(X, Y)
    #mlpCrossVal(X, Y)
    #svmCrossVal(X,Y)
    #kNeighborsCrossVal(X,Y)
    #pnnTrainTestNoNorm(X, Y)
    #pnnTrainTestNorm(X, Y)