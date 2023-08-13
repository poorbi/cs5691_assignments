import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

#Extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

w, h = 1, len(X)

x = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(X)):
    x[i][0] = X[i][0]

w, h = len(X), 1

y = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(X)):
    y[0][i]=X[i][1]

#Computing kernel matrix using given polynomial function n = 2
w, h = len(X), len(X)

K = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(X)):
    for j in range(len(X)):
        K[i][j] = (1+(x[i][0]*x[j][0] + y[0][i]*y[0][j]))**2

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

#T = I - 1n
w, h = len(X), len(X)

T = [[0 for x in range(w)] for y in range(h)]

for i in range(len(X)):
    for j in range(len(X)):
            T[i][j]=I[i][j]-n[i][j]

#Computing Kc = T.K.T where T = I - 1n for centering data
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

Kct = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(Kc)):
   for j in range(len(Kc[0])):
       Kct[i][j] = Kc[j][i]

#Plotting the data
Xpoints = np.dot(Kct,alpha1)
Ypoints = np.dot(Kct,alpha2)

plt.scatter(Xpoints,Ypoints,c='yellowgreen')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title("Polynomial Kernel PCA for Centered Data for degree = 2",color='black')
plt.grid()
plt.show()