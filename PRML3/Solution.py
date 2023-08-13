import pandas as pd
import numpy as np
import nltk
from nltk.corpus import words
import os
import csv
import ssl

# There was an error that came up using nltk regarding ssl so this code helps get over the error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Code to read the folder test
vec = []

# To get current directory 
path = os.getcwd()
# Retain this path for future use
oldpath = path
# Go to test folder by appending '/test' and changing directory
path = path+"/test"
os.chdir(path)

# Function that reads and appends the data of each text file in the test folder into a list
def read_text_file(file_path):
    with open(file_path, 'r') as f:
        vec.append([f.read()])

# Calling the read_text_file function for all .txt files in folder test 
for file in os.listdir():
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"
        read_text_file(file_path)

# Back to directory out of test
os.chdir(oldpath)

# Generating a csv file for test data using the list above for test emails
filename = "emails_test.csv"

with open(filename,'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerows(vec)

# Reading Training dataset
data = pd.read_csv('emails.csv',header=None,index_col=False)
dataArray = np.array(data)
m = len(dataArray)
n = len(dataArray[0])

# using nltk library for generating dictionary corresponding to training data
nltk.download('words')
sensible_words=set(words.words()) 
dictionary = []

for i in range(m):
  email = dataArray[i][0].split()
  size = len(email)
  word=""
  for j in range(size):
    word = email[j].lower()
    if word not in dictionary:
      if word in sensible_words:
        dictionary.append(word)

# Generating X and y by using dictionary created above and training dataset
# X has the size of number emails x length of dictionary
X = [[0 for k in range(len(dictionary))]for k in range(m)]
y = [0 for k in range(m)]

# setting 1 corresponding to words of dictionary that are present in that particular mail
for i in range(m):
  email = dataArray[i][0].split()
  size = len(email)
  word=""
  for j in range(size):
    word = email[j].lower()
    if word in dictionary:
      ind = dictionary.index(word)
      X[i][ind] = 1

for i in range(m):
  y[i] = dataArray[i][1]

# Generating X_test by using dictionary created above and testing data set
testdata = pd.read_csv('emails_test.csv',header=None,index_col=False)
testdataArray = np.array(testdata)

m_test = len(testdataArray)
X_test = [[0 for k in range(len(dictionary))]for k in range(m_test)]

# Setting 1 corresponding to words of dictionary that are present in that particular mail
for i in range(m_test):
  email = testdataArray[i][0].split()
  size = len(email)
  for j in range(size):
    word = email[j].lower()
    if word in dictionary:
      ind = dictionary.index(word)
      X_test[i][ind] = 1

# Funtion to calculate p hat which is the fraction of spam mails
def find_p_hat():
  count_spam_mails = 0
  sizey = len(y)
  for i in range(sizey):
    if y[i]==1:
      count_spam_mails+=1
  total_mails = len(dataArray)
  return count_spam_mails/total_mails

# Function to calculate the pjy's where p10 represents the number of non spam mails that contains 1st word of dictionary/number of non spam mails
def find_pjy():
  total_mails = len(dataArray)
  spam_mails = find_p_hat() * len(dataArray)
  pjy = [[0 for k in range(2)]for k in range(len(dictionary))]
  for j in range(len(dictionary)):
    count_y0 = 0
    count_y1 = 0
    for i in range(m):
      if X[i][j]==1:
        if y[i]==0:
          count_y0+=1
        else:
          count_y1+=1
    pjy[j][0] = count_y0/(total_mails-spam_mails)
    pjy[j][1] = count_y1/spam_mails
  return pjy

# Function to calculate predicted labels for test data
def find_ypred_test():
  pjy = find_pjy()
  p_hat = find_p_hat()
  yp = [0 for k in range(m_test)]
  for i in range(len(X_test)):
    sum = 0
    for j in range(len(X_test[0])):
      S1= 0
      S2 = 0
      if pjy[j][0]!=0 and pjy[j][1]!=0 :
        b = pjy[j][1]/pjy[j][0]
        S1 = (X_test[i][j]*np.log(b))
      if (1-pjy[j][1])!=0 and (1-pjy[j][0])!=0 :
        c = (1-pjy[j][1])/(1-pjy[j][0])
        S2 = ((1-X_test[i][j])*np.log(c))
      sum += S1 + S2 
    if(sum>=0):
      yp[i] = 1
    else:
      yp[i] = 0
  return yp

# Printing predicted labels for each mail in test data
yp = find_ypred_test()
print("Predicted Labels For Emails")

for i in range(len(yp)):
    print("email ",i+1," label = ",yp[i])

final_result = []

for i in range(len(yp)):
    final_result.append([vec[i][0],yp[i]])

# Storing the predicted labels for each mail in test data in a csv file named "emails_predicted_labels.csv"
file_name = "emails_predicted_labels.csv"

with open(file_name,'w') as csvfile:
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(['emails','predicted label'])
    csvwriter.writerows(final_result)
