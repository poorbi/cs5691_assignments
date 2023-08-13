from turtle import color
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import sys

#extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

Xt = X.T

#Computing kernel matrix using given radial function sigma s=0.9
col,row=len(X),len(X)
K1=[[0 for i in range(col)] for j in range(row)]
Yt=[[0 for i in range(1)] for j in range(2)]

s= 0.9
for i in range(row):
    for j in range(col):
        Y = X[i,:]-X[j,:]
        K1[i][j]=math.exp((-0.5/(s*s))*np.dot(Y,Y.T))

w, h = 1, len(K1)

Z = [[0 for s in range(w)] for t in range(h)] 

for i in range(len(Z)):
    for j in range(len(Z[0])):
        Z[i][j] = 0 

#Computing Identity matrix of 1000X1000
w, h = len(X), len(X)

I = [[0 for s in range(w)] for t in range(h)] 

for i in range(len(X)):
    for j in range(len(X)):
        if(i == j):
            I[i][j]=1

#Computing 1n matrix of 1000X1000 with all elements 1/n where n = 1000
w, h = len(X), len(X)

n = [[(1/len(X)) for s in range(w)] for t in range(h)] 

w, h = len(X), len(X)

#Computing T = I - 1n
T = [[0 for s in range(w)] for t in range(h)]

for i in range(len(X)):
    for j in range(len(X)):
            T[i][j]=I[i][j]-n[i][j]

#Computing Kc = T.K.T where T = I - 1n
w, h = len(X), len(X)

Kc = [[0 for s in range(w)] for t in range(h)] 

w, h = len(X), len(X)

Temp = [[0 for s in range(w)] for t in range(h)]

Temp = np.dot(T,K1) 
Kc = np.dot(Temp,T)

#Computing top two eigen values and vectors
eigenval, eigenvec = np.linalg.eig(Kc)

#New data set using the top 4 eigen vectors
index = eigenval.argsort()[::-1]
eigenval = eigenval[index]
eigenvec = eigenvec[:,index]

w, h = 4, len(Kc)

Evs =  [[0 for s in range(w)] for t in range(h)]

for i in range(len(Evs)):
    for j in range(len(Evs[0])):
        Evs[i][j]=eigenvec[i][j]

#Normalizing the rows of the new data set
for i in range(len(Evs)):
    n = np.linalg.norm(Evs[i])
    for j in range(len(Evs[0])):
        Evs[i][j]=Evs[i][j]/n

#Finding cluster using argmax of a row
for i in range(len(Evs)):
    P = Evs[i]
    for j in range(len(Evs[0])):
        max_value = max(P)
        ind = P.index(max_value)
        Z[i][0]=ind
    
colorlist = ['slateblue','plum','greenyellow','purple','yellow']

#Plotting clusters for new data set
for j in range(4):
        for i in range(len(X)):
            if Z[i][0]==j :
                plt.xlabel('x-axis',color='black')
                plt.ylabel('y-axis',color='black')
                plt.title('Raidal Kernel Cluster k = 4 with sigma = 0.9 (argmax)', color = 'black')
                plt.scatter(X[i][0],X[i][1],c = colorlist[j])

plt.grid()
plt.show()