import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import sys

#Extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

#Creating a third column in data X to store cluster value of each point
w,h=len(X[0])+1,len(X)

K=[[0 for i in range(w)] for j in range(h)]

for i in range(len(X)):
    for j in range(len(X[0])):
        K[i][j] = X[i][j]

for i in range(len(X)):
    K[i][2] = 0

#Random intialization by setting four random data points as centroids from given data
mX1=[]
mX2=[]
mX3=[]
mX4=[]

i1 = random.randint(0,len(K)-1)
mX1.append(K[i1][0])
mX1.append(K[i1][1])
i2 = random.randint(0,len(K)-1)
mX2.append(K[i2][0])
mX2.append(K[i2][1])
i3 = random.randint(0,len(K)-1)
mX3.append(K[i3][0])
mX3.append(K[i3][1])
i4 = random.randint(0,len(K)-1)
mX4.append(K[i4][0])
mX4.append(K[i4][1])

#Flag to check if reassignment of cluster has taken place in an iteration or not for convergence
check = 1

it = 0
iter = []
errorSum = []

#Iterations for reassignment of means to data points
while(check):

    check = 0

    for i in range(len(K)):

        z = K[i][2]
        mindist = sys.maxsize

        #Finding minimum distance of every point to the means and assigning it that cluster

        if(len(mX1)!=0 ):
            if (((K[i][0]-mX1[0])**2)+((K[i][1]-mX1[1])**2))**0.5 < mindist:
                mindist = (((K[i][0]-mX1[0])**2)+((K[i][1]-mX1[1])**2))**0.5
                z = 1
        if(len(mX3)!=0 ):
            if (((K[i][0]-mX3[0])**2)+((K[i][1]-mX3[1])**2))**0.5 < mindist:
                mindist = (((K[i][0]-mX3[0])**2)+((K[i][1]-mX3[1])**2))**0.5
                z = 3
        if(len(mX4)!=0 ):
            if (((K[i][0]-mX4[0])**2)+((K[i][1]-mX4[1])**2))**0.5 < mindist:
                mindist = (((K[i][0]-mX4[0])**2)+((K[i][1]-mX4[1])**2))**0.5
                z = 4
        if(len(mX2)!=0 ):
            if (((K[i][0]-mX2[0])**2)+((K[i][1]-mX2[1])**2))**0.5 < mindist:
                mindist = (((K[i][0]-mX2[0])**2)+((K[i][1]-mX2[1])**2))**0.5
                z = 2
            
        if(K[i][2]!=z):
            K[i][2]=z
            check = 1
    
    #Recalculating means of all clusters after resassigment step

    X1 =[]
    X2 =[]
    X3 =[]
    X4 =[]

    for i in range(len(K)):
        if(K[i][2]==1):
            X1.append([K[i][0],K[i][1]])

    for i in range(len(K)):
        if(K[i][2]==2):
            X2.append([K[i][0],K[i][1]])

    for i in range(len(K)):
        if(K[i][2]==3):
            X3.append([K[i][0],K[i][1]])

    for i in range(len(K)):
        if(K[i][2]==4):
            X4.append([K[i][0],K[i][1]])

    mX1 = [float(sum(l))/len(l) for l in zip(*X1)]
    mX2 = [float(sum(l))/len(l) for l in zip(*X2)]
    mX3 = [float(sum(l))/len(l) for l in zip(*X3)]
    mX4 = [float(sum(l))/len(l) for l in zip(*X4)]

    #Calculation error for every iteration and adding it to a list
    sumE = 0
    for i in range(len(K)):
        if(K[i][2]==1):
            if(len(mX1)!=0):
                 sumE += (((K[i][0]-mX1[0])**2)+((K[i][1]-mX1[1])**2))**0.5
        if(K[i][2]==2):
            if(len(mX1)!=0):
                 sumE += (((K[i][0]-mX2[0])**2)+((K[i][1]-mX2[1])**2))**0.5
        if(K[i][2]==3):
            if(len(mX1)!=0):
                 sumE += (((K[i][0]-mX3[0])**2)+((K[i][1]-mX3[1])**2))**0.5
        if(K[i][2]==4):
            if(len(mX1)!=0):
                 sumE += (((K[i][0]-mX4[0])**2)+((K[i][1]-mX4[1])**2))**0.5

    it+=1
    iter.append(it)
    errorSum.append(sumE)

#Plotting the data
f = plt.figure(1)
for i in range(len(K)):
    if K[i][2]==1 :
        plt.scatter(K[i][0],K[i][1],c = 'slateblue')
    if K[i][2]==2:
        plt.scatter(K[i][0],K[i][1],c = 'plum')
    if K[i][2]==3 :
        plt.scatter(K[i][0],K[i][1],c = 'greenyellow')
    if K[i][2]==4 :
        plt.scatter(K[i][0],K[i][1],c = 'purple')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("K-means Clustering for k=4",color='black')
plt.grid()

#Plottig the error function
g = plt.figure(2)
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Error Function",color='black')
plt.grid()
plt.plot(iter,errorSum)

plt.show()