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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import FastICA
# Needed to install neupy (sudo pip install neupy)
from sklearn import datasets, metrics
#from neupy import algorithms, environment

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

# Needed to install  imbalanced-learn (sudo pip install sudo pip install -U imbalanced-learn)
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import ADASYN, SMOTE

# Not working!
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE

RANDOM_STATE = 42
# Number of components after dimennsional reduction
K =60

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
Apparently non-interesting fields:
    - "Arboles_Metros_Calzada", 
    - "Colision_Vehiculo_Obstaculo_1",
    - "Colision_Vehiculo_Obstaculo_2", 
    - "Dia", 
    - "Hora",
    - "Hm"      65% (without these)
    - "Sentido"
    - "GPS_z",
    - "IMD"
"""
def analyze_select_data(data, train=True):
    
    Y = []
    if train:
        # If there are any dead or injured -> class 1
        Y_labels = ["N_Muertos", "N_Graves", "N_Leves"]
        for val in data[Y_labels].values:
            if val.any() > 0:
                Y.append(1)
            else:
                Y.append (0)
        Y_labels = ["N_Muertos", "N_Graves", "N_Leves"]
        # Checking if the class are balanced
        print("P Class 1 (Y):",  float(Y.count(1))/len(Y))
        print("P Class 0 (Y):",  float(Y.count(0))/len(Y))    
    
        X = data.drop(Y_labels, axis=1)
    else:
        X = data
    
    # Remove colums with no interest and srting values and nan values
    rm_cols = ["Id", "Numero_Accidente", "Carretera", "Pk", "Km", "Tipo_Est"]
    more_rm_cols = ["Arboles_Metros_Calzada", "Colision_Vehiculo_Obstaculo_1",
                "Colision_Vehiculo_Obstaculo_2", "Dia", "Hora", "Hm",   #65
                "Sentido", "GPS_z", "IMD"]
    rm_cols += more_rm_cols
    X.drop(rm_cols, axis=1, inplace=True)
    X.fillna(value=0, inplace=True)
    # print_all_data(X,Y) 
    # print_column("Arboles_Metros_Calzada", X, Y)

    return X, Y


"""
    MLP network with the 80% if data for training and 20% for validation
"""
def mlpSimpleDiv(X, Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)
    X_red_train = FastICA(n_components=K, whiten=True).fit_transform(X_train, Y)
    X_red_val = FastICA(n_components=K, whiten=True).fit_transform(X_val, Y)
    
    # Normalization
    scalar = StandardScaler()
    X_train_n = scalar.fit_transform(X_train)
    X_val_n = scalar.fit_transform(X_val)
    
    # MLP creation + trainning
    scalar = StandardScaler()
    mlp = MLPClassifier(activation="identity", verbose=False, solver="adam",
                        max_iter=150, hidden_layer_sizes=(100,3), early_stopping=True, 
                        tol=1e-12, validation_fraction=0.3, alpha=1e-8,
                        learning_rate_init=0.1, beta_1=0.5, warm_start=True,
                        random_state=RANDOM_STATE)
                        
    undersample = SMOTE()
    classifier = make_pipeline(undersample, mlp)
    # Prediction and evaluation
    classifier.fit(X_train_n, Y_train)
    prediction = classifier.predict(X_val_n)
    print ("\n", classification_report(prediction, Y_val))
    return classifier, scalar


"""
    MLP network using cross validation for evaluating the model
    
    Best config (66 +- 2)
    Using X
    mlp = MLPClassifier(activation="identity", verbose=True,
                        max_iter=150, hidden_layer_sizes=20, early_stopping=False, 
                        tol=1e-8, validation_fraction=0.3, alpha=1e-10, 
                        learning_rate_init=0.0001, beta_1=0.8, warm_start=True) 
"""
def mlpCrossVal(X, Y):
    X_red = FastICA(n_components=K, whiten=True).fit_transform(X,Y) #65
    #X_red = NMF(n_components=30).fit_transform(X, Y)
    #X_red = TruncatedSVD(n_components=55).fit_transform(X, Y)          #65 +- 4
    #X_red = SelectKBest(mutual_info_classif, k=30).fit_transform(X, Y)

    
    
    scalar = StandardScaler()
    #undersample = NearMiss(version=3, random_state=RANDOM_STATE)
    undersample = SMOTE()
    
    mlp = MLPClassifier(activation="identity", verbose=False, solver="adam",
                        max_iter=150, hidden_layer_sizes=(400,3), early_stopping=True, 
                        tol=1e-12, validation_fraction=0.3, alpha=1e-8,
                        learning_rate_init=0.1, beta_1=0.5, warm_start=True,
                        random_state=RANDOM_STATE)    
    classifier = make_pipeline(undersample, mlp)
    pipeline = make_pipeline(scalar, classifier)
    scores = cross_val_score(pipeline, X_red, Y, cv=3)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return classifier, scalar
    

# Mejor precison media:     0.80      0.62      0.66
# Con pesos balanceados:    0.76      0.62      0.66
def svmSimpleVal(X,Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)
   # X_red_train = FastICA(n_components=K, whiten=True).fit_transform(X_train, Y)
    #X_red_val = FastICA(n_components=K, whiten=True).fit_transform(X_val)
    
    # Normalization
    scalar = StandardScaler()
    X_train_n = scalar.fit_transform(X_train)
    X_val_n = scalar.fit_transform(X_val)
    
    undersample = SMOTE()
    svm = SVC(verbose=True, kernel="rbf", decision_function_shape="ovr",
              random_state=RANDOM_STATE, C=0.008, degree=2)
              
    classifier = make_pipeline(undersample, svm)
    
    # Prediction and evaluation
    classifier.fit(X_train_n, Y_train)
    prediction = classifier.predict(X_val_n)
    print ("\n", classification_report(prediction, Y_val))
    return classifier, scalar


def svmCrossVal(X, Y):
    X_red = FastICA(n_components=K, whiten=True).fit_transform(X, Y)
    
    scalar = StandardScaler()
    undersample = SMOTE()
    svm = SVC(verbose=True, kernel="rbf", decision_function_shape="ovr",
              random_state=RANDOM_STATE, C=0.008, degree=2,
              class_weight="balanced")
    classifier = make_pipeline(undersample, svm)
    pipeline = make_pipeline(scalar, classifier)
    scores = cross_val_score(pipeline, X, Y, cv=5)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return svm, scalar


def kNeighborsCrossVal(X, Y):    
    scalar = StandardScaler()
    kn = KNeighborsClassifier(n_neighbors=101, p=2, weights="distance")
    pipeline = make_pipeline(scalar, kn)
    scores = cross_val_score(pipeline, X, Y, cv=5)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return kn, scalar
    

def kNeighborsSimpleVal(X,Y):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, train_size=0.8)
   # X_red_train = FastICA(n_components=K, whiten=True).fit_transform(X_train, Y)
    #X_red_val = FastICA(n_components=K, whiten=True).fit_transform(X_val)
    
    # Normalization
    scalar = StandardScaler()
    X_train_n = scalar.fit_transform(X_train)
    X_val_n = scalar.fit_transform(X_val)
    
    undersample = SMOTE()
    kn = KNeighborsClassifier(n_neighbors=3001, p=2, weights="uniform",
                              algorithm="auto")
              
    classifier = make_pipeline(undersample, kn)
    
    # Prediction and evaluation
    classifier.fit(X_train_n, Y_train)
    prediction = classifier.predict(X_val_n)
    print ("\n", classification_report(prediction, Y_val))
    return classifier, scalar


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


def logisticRegr(X,Y):
    X_red = FastICA(n_components=50, whiten=True).fit_transform(X, Y)
    scalar = LogisticRegression()   
    lr = LogisticRegression()
    pipeline = make_pipeline(scalar, lr)
    scores = cross_val_score(pipeline, X_red, Y, cv=8)

    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores, lr


"""
    The selected classfier fits all the train data.
    As we are using more data, the accuracy should be better than the obtained
    before.
    Then, we read the test data, remove the colums we have ignored for the classification
    and predict the results.
"""
def predictTestData(X_train, Y_train, classifier, scalar):
    #X_red_train = FastICA(n_components=K, whiten=True).fit_transform(X_train)
    Xn = scalar.fit_transform(X_train)
    classifier.fit(Xn, Y_train)
    test_data = pd.read_csv('testdata.csv')
    X_test, Y_test = analyze_select_data(test_data, False)
    #X_red_test = FastICA(n_components=K, whiten=True).fit_transform(X_test)
    Xn_test = scalar.fit_transform(X_test)
    pred = pd.DataFrame(classifier.predict(Xn_test),index=np.arange(1,201))#index=np.arange(1,201)
    pred.to_csv("predicted.csv", header=["Prediction"], index_label="Id")
    #print (classifier.score(Xn_test, Y_test))
    print ("DONE")


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
    
    #classifier, scalar = mlpSimpleDiv(X, Y)
    #classifier, scalar = mlpCrossVal(X, Y)
    #logisticRegr(X,Y)
    #classifier, scalar = svmCrossVal(X,Y)
    classifier, scalar = svmSimpleVal(X,Y)
    #classifier, scalar = kNeighborsCrossVal(X,Y)
    #clasifier, scalar = kNeighborsSimpleVal(X,Y)
    #pnnTrainTestNoNorm(X, Y)
    #pnnTrainTestNorm(X, Y)
    
    predictTestData(X, Y, classifier, scalar)