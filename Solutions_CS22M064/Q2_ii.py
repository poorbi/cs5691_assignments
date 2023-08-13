import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import sys

#extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

w,h=len(X[0])+1,len(X)

K=[[0 for i in range(w)] for j in range(h)]

for i in range(len(X)):
    for j in range(len(X[0])):
        K[i][j] = X[i][j]

for i in range(len(X)):
    K[i][2] = 0

mX = []
for i in range(5):
    i1 = random.randint(0,len(K)-1)
    l =[]
    for m in range(2):
           l.append(K[i1][m])
    mX.append(l)

def plotClusters(k):

    check = 1

    it = 0
    iter = []
    errorSum = []

    while(check):

        check = 0

        for i in range(len(K)):
            
            z = K[i][2]
            mindist = sys.maxsize

            for j in range(k):
                if(len(mX[j])!=0):
                    if (((K[i][0]-mX[j][0])**2)+((K[i][1]-mX[j][1])**2))**0.5 < mindist:
                        mindist = (((K[i][0]-mX[j][0])**2)+((K[i][1]-mX[j][1])**2))**0.5
                        z = j
                
            if(K[i][2]!=z):
                K[i][2]=z
                check = 1

        sumE = 0
        for i in range(len(K)):
            for s in range(k):
                if(K[i][2]==s):
                    if(len(mX[s])!=0):
                        sumE += (((K[i][0]-mX[s][0])**2)+((K[i][1]-mX[s][1])**2))**0.5

        it+=1
        iter.append(it)
        errorSum.append(sumE)
        
        for i in range(k):

            X=[]

            for j in range(len(K)):
                if(K[j][2]==i):
                    X.append([K[j][0],K[j][1]])

            mX[i]=[float(sum(l))/len(l) for l in zip(*X)]

    colorlist = ['slateblue','plum','greenyellow','purple','yellow']

    f = plt.figure(1)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("K-Means Clustering for k = "+str(k),color='black')
    plt.grid()
    for j in range(k):
        for i in range(len(K)):
            if K[i][2]==j :
                plt.scatter(K[i][0],K[i][1],c = colorlist[j])


    g = plt.figure(2)
    plt.grid()
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("Error Function",color='black')
    plt.plot(iter,errorSum)

    h=plt.figure(3)
    i = -11
    z = 0
    while(i<11.0):
        j=-11
        #print(i)
        while(j<11.0):
            #print(i)
            mindist = sys.maxsize
            for q in range(k):
                if(len(mX[q])!=0):
                    if (((i-mX[q][0])**2)+((j-mX[q][1])**2))**0.5 < mindist:
                        mindist = (((i-mX[q][0])**2)+((j-mX[q][1])**2))**0.5
                        z = q
            plt.scatter(i,j,c=colorlist[z])
            j+=0.5
        i+=0.5
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title("Voronoi Regions of K-Means Clustering for k = "+str(k),color='black')
    plt.show()


for i in range(2,6):
    plotClusters(i)