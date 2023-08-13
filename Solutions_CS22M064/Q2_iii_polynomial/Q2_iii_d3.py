from turtle import color
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import sys

#Extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

w, h = 1, len(X)

x = [[0 for s in range(w)] for t in range(h)] 

for i in range(len(X)):
    x[i][0] = X[i][0]

w, h = len(X), 1

y = [[0 for s in range(w)] for t in range(h)] 

for i in range(len(X)):
    y[0][i]=X[i][1]

#Computing kernel matrix using given function for degree = 3
w, h = len(X), len(X)

K1 = [[0 for s in range(w)] for t in range(h)] 

for i in range(len(X)):
    for j in range(len(X)):
        K1[i][j] = (1+(x[i][0]*x[j][0] + y[0][i]*y[0][j]))**3

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

#Plotting clusters for new data set
def plotClusters(k):

    #Random intialization by setting k random data points as centroids from given data
    mX = []
    for i in range(k):
        i1 = random.randint(0,len(Evs)-1)
        l =[]
        for m in range(4):
            l.append(Evs[i1][m])
        mX.append(l)

    check = 1

    it = 0
    iter = []
    errorSum = []

    #Iterations for reassignment of means to data points
    while(check):

        check = 0

        for i in range(len(Evs)):
            
            z = Z[i][0]
            mindist = sys.maxsize

            #Finding minimum distance of every point to the means and assigning it that cluster
            for j in range(k):
                if(len(mX[j])!=0):
                    if (((Evs[i][0]-mX[j][0])**2)+((Evs[i][1]-mX[j][1])**2)+((Evs[i][2]-mX[j][2])**2)+((Evs[i][3]-mX[j][3])**2))**0.5 < mindist:
                        mindist = (((Evs[i][0]-mX[j][0])**2)+((Evs[i][1]-mX[j][1])**2)+((Evs[i][2]-mX[j][2])**2)+((Evs[i][3]-mX[j][3])**2))**0.5
                        z = j
                
            if(Z[i][0]!=z):
                Z[i][0]=z
                check = 1
        
        #Recalculating means of all clusters after resassigment step
        for i in range(k):

            XMean=[]

            for j in range(len(Evs)):
                if(Z[j][0]==i):
                    XMean.append([Evs[j][0],Evs[j][1],Evs[j][2],Evs[j][3]])

            mX[i]=[float(sum(l))/len(l) for l in zip(*XMean)]
        
        #Calculation error for every iteration and adding it to a list
        sumE = 0
        for i in range(len(Evs)):
            for s in range(k):
                if(Z[i][0]==s):
                    if(len(mX[s])!=0):
                        sumE += (((Evs[i][0]-mX[s][0])**2)+((Evs[i][1]-mX[s][1])**2)+((Evs[i][2]-mX[s][2])**2)+((Evs[i][3]-mX[s][3])**2))**0.5

        it+=1
        iter.append(it)
        errorSum.append(sumE)


    #Plotting the data
    colorlist = ['slateblue','plum','greenyellow','purple','yellow']


    f = plt.figure(1)
    for j in range(k):
        for i in range(len(X)):
            if Z[i][0]==j :
                plt.xlabel('x-axis',color='black')
                plt.ylabel('y-axis',color='black')
                plt.title('Polynomial Kernel Cluster k = 4 and degree = 3', color = 'black')
                plt.scatter(X[i][0],X[i][1],c = colorlist[j])
    plt.grid()

    g = plt.figure(2)
    plt.xlabel('x-axis',color='black')
    plt.ylabel('y-axis',color='black')
    plt.title('Error Function', color = 'black')
    plt.grid()
    plt.plot(iter,errorSum)

    plt.show()
    input()

plotClusters(4)
