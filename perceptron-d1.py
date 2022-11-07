import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import pdb


#importing dataset
dataset=pd.read_csv('costpd.csv')
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dataset = dataset.apply(le.fit_transform)
data = dataset.to_numpy()
X = data[:, :-1] #input features 
y = data[:,-1] #output 

#Splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
#feature scaling(similar to normalizing)
from sklearn.preprocessing import StandardScaler
# from sklearn import preprocessing
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# data prep
X_train = torch.from_numpy(X_train)
X_train = X_train.type(torch.double) 
Y_train = torch.from_numpy(y_train)
Y_train = Y_train.type(torch.double) 
print('requires_grad for X_train: ', X_train.requires_grad)
print('requires_grad for Y_train: ', Y_train.requires_grad)


input_size = X_train.shape[1] 
hidden_size = 1
output_size = 1 
learning_rate = 0.001
w1 = torch.rand(input_size,hidden_size,requires_grad=True)
w1 = w1.type(torch.double)
b1 = torch.rand(hidden_size,output_size,requires_grad=True)
b1 = b1.type(torch.double)

for iter in range(1, 4001):
    pdb.set_trace()
    y_pred = X_train.mm(w1).clamp(min=0).add(b1)
    loss = (y_pred - Y_train).pow(2).sum() 
    if iter % 100 ==0:
        print(iter, loss.item())
    loss.backward()
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w1.grad.zero_()
        b1.grad.zero_()

print ('w1: ', w1)
print ('b1: ', b1)

predicted_in_tensor = X_train_tensor.mm(w1).clamp(min=0).add(b1)

plt.figure(figsize=(8, 8))
plt.scatter(x_train, y_train, c='green', s=200, label='Original data')
plt.plot(x_train, predicted, label = 'Fitted line')
plt.legend()
plt.show()