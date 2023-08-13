import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Extracting data from the csv file into a matrix X. Here data is not centered
data = pd.read_csv('Dataset.csv', header=None, index_col=False)
X = np.array(data)

#Plotting the data from matrix X (data matrix)
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


#Finding eigen values and vectors of covarianve matrix C
eigenval, eigenvec = np.linalg.eig(C)

#Sorting eigen values in descending order and finding vectors corresponding to them
index = eigenval.argsort()[::-1]
eigenval = eigenval[index]
eigenvec = eigenvec[:,index]

#Printing the variance for data along different components in descending order
for i in range(2):
    variance=eigenval[i]/eigenval.sum()
    print("variance along component" ,i+1," = ", round(variance.real*100,4),"%")

#Plotting lines correspondinh to the eigen vectors (top two)
plt.axline((0,0),(-0.323516, -0.9462227),color = "purple",label = 'w1')
plt.axline((0,0),(-0.9462227, 0.323516),color = "black",label = 'w2')
plt.title("PCA for Uncentered Data",color='black')
plt.legend()
plt.grid()
plt.show()