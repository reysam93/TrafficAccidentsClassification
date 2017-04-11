# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 19:26:57 2017

@author: samuel
"""

import pandas as pd

dataset = pd.read_csv('traindata.csv')

cont = 0
for col in dataset["N_Muertos"]:
    if col > 0:
        cont += 1
 
print(cont)
print(len(dataset["N_Muertos"]))
print(cont/len(dataset["N_Muertos"]))
