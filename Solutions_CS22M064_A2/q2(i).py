# -*- coding: utf-8 -*-
"""Q2(i).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vwjTrOFifCXwbknACwGgos__1hxuTr-p
"""

#Importing required modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting data from the given csv files 
train_data = pd.read_csv('A2Q2Data_train.csv', header=None, index_col=False)
test_data = pd.read_csv('A2Q2Data_test.csv', header=None, index_col=False)
a = np.array(train_data)
t = np.array(test_data)

#Extracting all X values from training dataset
rows,cols = (len(a),(len(a[0])-1))

X = []
for i in range(rows):
  col = []
  for j in range(cols):
    col.append(0)
  X.append(col)

#Extracting all y values from training dataset

rowsy,colsy = (len(a),1)
y = []
for i in range(rowsy):
  coly = []
  for j in range(colsy):
    coly.append(0)
  y.append(coly)

for i in range(len(X)):
  for j in range(len(X[0])):
    X[i][j] = a[i][j]

for i in range(len(a)):
  y[i][0] = a[i][100]

#Converting X to array from list of lists
X = np.array(X)

#Finding X transpose
Xt = X.transpose()

#Applying ((XtX)^-1)(Xty) to find w^ml
C = np.dot(Xt,X) 

Ci = np.linalg.inv(C)

w = np.dot(np.dot(Ci,Xt),y)

#Extracting all X values from testing dataset
rows,cols = (len(t),(len(t[0])-1))

T = []
for i in range(rows):
  col = []
  for j in range(cols):
    col.append(0)
  T.append(col)

rowsy,colsy = (len(t),1)

#Extracting all y values from testing dataset

yt = []
for i in range(rowsy):
  coly = []
  for j in range(colsy):
    coly.append(0)
  yt.append(coly)

for i in range(len(T)):
  for j in range(len(T[0])):
    T[i][j] = t[i][j]

for i in range(len(t)):
  yt[i][0] = t[i][100]

#Finding predicted y of testing data set using calculated w^ml

yn = np.dot(T,w)

print(yn)

#Printing Error percent over testing dataset
sum = 0
for i in range(len(T)):
  sum += ((yn[i][0] - yt[i][0])**2)

error = sum**0.5

print(error)

#Printing Accuracy percent over testing dataset
sumy=0

for i in range(len(T)):
  sumy += yt[i][0]**2

ynorm = sumy**0.5

acc = ((ynorm - error)/ ynorm)*100

print(acc)