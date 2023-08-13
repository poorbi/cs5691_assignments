import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

Xt = X.T

#Computing kernel matrix using given radial function for s = 0.1
col,row=len(X),len(X)
K=[[0 for i in range(col)] for j in range(row)]
Yt=[[0 for i in range(1)] for j in range(2)]

s = 0.1
for i in range(row):
    for j in range(col):
        Y = X[i,:]-X[j,:]
        K[i][j]=math.exp((-0.5/(s*s))*np.dot(Y,Y.T))

#Computing Identity matrix of 1000X1000
w, h = len(X), len(X)

I = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(X)):
    for j in range(len(X)):
        if(i == j):
            I[i][j]=1

#Computing 1n matrix of 1000X1000 with all elements 1/n where n = 1000
w, h = len(X), len(X)

n = [[(1/len(X)) for x in range(w)] for y in range(h)] 

w, h = len(X), len(X)

#T = I - 1n
T = [[0 for x in range(w)] for y in range(h)]

for i in range(len(X)):
    for j in range(len(X)):
            T[i][j]=I[i][j]-n[i][j]

#Computing Kc = T.K.T where T = I - 1n
w, h = len(X), len(X)

Kc = [[0 for x in range(w)] for y in range(h)] 

w, h = len(X), len(X)

Temp = [[0 for x in range(w)] for y in range(h)]

Temp = np.dot(T,K) 
Kc = np.dot(Temp,T)

eigenval, eigenvec = np.linalg.eig(Kc)

#Computing top two eigen values and vectors
eigenvalue1 = eigenval[0]
eigenvalue2 = eigenval[1]

eigenvector1 = eigenvec[:,0]
eigenvector2 = eigenvec[:,1]

#Computing alpha for the top two eigen values
c1 = 1/math.sqrt(eigenvalue1)
c2 = 1/math.sqrt(eigenvalue2)

alpha1 = c1 * eigenvec[:,0]
alpha2 = c2 * eigenvec[:,1]

#Computing transpose of the centered kernel matrix
w, h = len(X), len(X)

Kt = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(K)):
   for j in range(len(K[0])):
       Kt[j][i] = K[i][j]

Xpoints = np.dot(Kt,alpha1)
Ypoints = np.dot(Kt,alpha2)

#Plotting the data
plt.scatter(Xpoints,Ypoints,c='yellowgreen')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Radial Kernel PCA for Centered Data for sigma = 0.1",color='black')
plt.grid()
plt.show()