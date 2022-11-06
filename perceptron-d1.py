import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

#importing dataset
dataset=pd.read_csv('costpd.csv')
data = dataset.to_numpy()
X = data[:, :-1] #input features 
y = data[:,-1] #output 

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#feature scaling(similar to normalizing)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

plt.figure(figsize=(8,8))
plt.scatter(x_train, y_train, c='green', s=200, label='Original data')
plt.show()