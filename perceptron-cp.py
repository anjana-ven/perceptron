from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, sampler, random_split
import torch
from torch.nn import *
from torch import nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import random
import seaborn as sns
import os

plt.switch_backend('agg')
sns.set()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Set training variables

batch_size = 16
learning_rate = 0.01
momentum = 1
training_epochs = 1000
lossfn = "MSELoss"
decay_rate = 1e-7
dataset = "costpd.csv"


class MyDataset(Dataset):
  def __init__(self, root):
      df = pd.read_csv(root, delimiter=",", header=0)
      le = LabelEncoder()
      df = df.apply(le.fit_transform)
      self.data = df.to_numpy()
      self.x = self.data[:, :-1]
      self.y = self.data[:, -1]
      self.length = self.data.shape[0]

  def __getitem__(self, idx):
    return torch.tensor(self.x[idx, :]), torch.tensor(self.y[idx])

  def __len__(self):
    return self.length

  def numDataFeatures(self):
    return self.x.shape[1]

class MLP(torch.nn.Module):

  def __init__(self, num_features, dim_hidden, numExtraLayers=3, outputdim=1):
    super().__init__()

    self.outputdim = outputdim

    ### 1st hidden layer
    self.linear_relu_1 = nn.Sequential(
                    nn.Linear(num_features, dim_hidden),
                    nn.ReLU()
    )
    self.extra_linear_relu_stack = []

    for layerNum in range(numExtraLayers):
      self.extra_linear_relu_stack.append(nn.Linear(dim_hidden, dim_hidden))
      self.extra_linear_relu_stack.append(nn.ReLU())

    self.extra_linear_relu_stack = nn.Sequential(*self.extra_linear_relu_stack)
    self.outputLayer = nn.Linear(dim_hidden, outputdim)

  def forward(self, x):
    x = self.linear_relu_1(x)
    x = self.extra_linear_relu_stack(x)
    x = self.outputLayer(x)
    return x


def train_one_epoch(model, lossfxn, optimizer, loader, e):
  model.train()
  total_loss = 0

  for batch_idx, (data, target) in enumerate(loader):
    data = data.to(device)
    data = data.to(torch.float32)
    # data = data[:,None,:]
    target = target.to(device)
    target = target.to(torch.float32)
    optimizer.zero_grad()
    # pdb.set_trace()
    output = model(data)
    loss = lossfxn(output, target)
    loss.backward()
    total_loss += loss.item()*len(data)

    optimizer.step()

    if batch_idx % batch_size == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        e, batch_idx * len(data), len(loader.dataset),
        100. * batch_idx / len(loader), loss.item()))

  # pdb.set_trace()

  avgTrainLoss = total_loss/len(loader.dataset)
  print('Train Epoch: %s - Avg. Training Loss: %f' % (e, avgTrainLoss))

  return avgTrainLoss


def validate(model, lossfxn, loader, e):
  model.eval()
  with torch.no_grad():
    total_loss = 0
    # classInfoDict = {}
    for batch_idx, (data, target) in enumerate(loader):
      data = data.to(device)
      data = data.to(torch.float32)
      target = target.to(device)
      target = target.to(torch.float32)
      output = model(data)
      # pdb.set_trace()
      loss = lossfxn(output, target)
      # pdb.set_trace()
      total_loss += loss.item()*len(data)
    avgValLoss = total_loss/len(loader.dataset)
    print('Train Epoch: %s - Avg. Validation Loss : %f' % (e, avgValLoss))

    return avgValLoss


def train():

  data = MyDataset(dataset)

  dataTrain, dataVal = random_split(data, [0.7, 0.3])

  train_loader = DataLoader(
    dataset=dataTrain, batch_size=batch_size, shuffle=True)

  val_loader = DataLoader(
    dataset=dataVal, batch_size=batch_size, shuffle=True)

  model = MLP(num_features=data.numDataFeatures(), dim_hidden=50).to(device)
  lossfxn = globals()[lossfn]()

  # pdb.set_trace()
  optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay_rate)
  train_loss = []
  val_loss = []

  for e in range(training_epochs):
    avgTrainLoss = train_one_epoch(
      model, lossfxn, optimizer, train_loader, e)
    train_loss.append(avgTrainLoss)
    avgValLoss = validate(model, lossfxn, val_loader, e)
    val_loss.append(avgValLoss)

    # scheduler.step(avgValLoss) # Check this. Figure out what needs to be done.
    train_plot = sns.lineplot(x=[x for x in range(e+1)],
                              y=train_loss, label="training")
    val_plot = sns.lineplot(x=[x for x in range(e+1)],
                            y=val_loss, label="validation")
    train_plot.set(xlabel="Epochs", ylabel=lossfn)
    plt.savefig(lossfn+"-"+str(batch_size)+"-"+str(learning_rate) +
                "-"+str(training_epochs)+".pdf", bbox_inches='tight')
    plt.clf()
    plt.close()
    
if __name__ == '__main__':
    train()