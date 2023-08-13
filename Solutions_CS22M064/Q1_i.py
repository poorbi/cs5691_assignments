import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting data from the csv file into a matrix x 
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
x = np.array(data)

#Finding the mean of data for centering
mean = [float(sum(l))/len(l) for l in zip(*x)]

#Finding matrix X after centering data of matrix x
X = x - mean

#Plotting the data from matrix x (data matrix)
plt.scatter(X[:,0],X[:,1],c='yellowgreen')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

#Finding transpose of X
w, h = len(X), len(X[0])

XT = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(X)):
   for j in range(len(X[0])):
       XT[j][i] = X[i][j]

#Finding Covariance Matrix C = X.XT
w, h = 2, 2

C = [[0 for x in range(w)] for y in range(h)] 

for i in range(len(XT)):
    for j in range(len(X[0])):
        for k in range(len(X)):
            C[i][j] += XT[i][k] * X[k][j]

#Finding eigen values and vectors of covariance matrix C
eigenval, eigenvec = np.linalg.eig(C)

#Sorting eigen values in descending order and finding vectors corresponding to them
index = eigenval.argsort()[::-1]
eigenval = eigenval[index]
eigenvec = eigenvec[:,index]

for i in range(len(eigenval)):
    eigenval[i] = eigenval[i]/len(X)

#Printing the variance for data along different components in descending order
for i in range(2):
    variance=eigenval[i]/eigenval.sum()
    print("variance along component" ,i+1," = ", round(variance.real*100,4),"%")

#Plotting lines corresponding to the eigen vectors (top two)
plt.axline((0,0),(-0.323516, -0.9462227),color = "purple",label = 'w1')
plt.axline((0,0),(-0.9462227, 0.323516),color = "black",label = 'w2')
plt.title("PCA for Centered Data",color='black')
plt.legend()
plt.grid()
plt.show()