#!/usr/bin/env python
# coding: utf-8

# In[42]:


import numpy as np
import torchvision
import torch

from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
from matplotlib import pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from PIL import Image
import re
from sklearn import preprocessing
import data
from data import *


# In[43]:


#df = pd.read_csv('./combined.csv', header='infer')
df = pd.read_csv('../../topology/merged.csv', header='infer')
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
df.drop_duplicates(subset="row", inplace = True)

# In[44]:


pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')
df['year'] = 0
df['year'] = df['label'].str.extract(pattern).astype('int64')
cols_to_norm = df.columns[4:-2]
df[cols_to_norm] = df[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


# In[87]:


class SimpleDNN(torch.nn.Module):
        
    #Our batch shape for input x is (3, 256, 256)
    
    def __init__(self, in_shape):
        super(SimpleDNN, self).__init__()
        self.in_shape = in_shape  
        self.fc1 = torch.nn.Linear(in_shape, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 16)
        self.fc4 = torch.nn.Linear(16, 2)
        self.dropout1 = torch.nn.Dropout(0.01)
        self.dropout2 = torch.nn.Dropout(0.001)
        self.dropout3 = torch.nn.Dropout(0.0001)
        
    def forward(self, x):
        
        x = x.view(-1, self.in_shape)    
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        
        return x


# In[88]:


import torch.optim as optim

def createLossAndOptimizer(net, learning_rate, weight):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    return(loss, optimizer)


# In[89]:


def get_loader(set, sampler, batch_size):    
    sunspotLoader = torch.utils.data.DataLoader(set, sampler=sampler, num_workers=2, batch_size=batch_size, shuffle=True)
    
    return sunspotLoader


# In[105]:


from importlib import reload
reload(data)

# Topological features
#cols = range(3,19)

# HMI features
#cols = range(19,38)

# All features
cols = range(4,38)

sunspotTrainSet, sunspotValidSet = data.generateTrainValidData(df, cols=cols, splitType='by_harpnum')
#sunspotTrainSet = sunspotPropertiesDataset(df, transform=None, isTrain=True)
#sunspotValidSet = sunspotPropertiesDataset(df, transform=None, isTrain=False)
#print(sunspotTrainSet.prop_frame.columns)
print(len(sunspotTrainSet), len(sunspotValidSet))
print(sunspotTrainSet.prop_frame.columns)


# In[106]:


import time

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_loader(sunspotTrainSet, sampler=None, batch_size=batch_size)
    val_loader = get_loader(sunspotValidSet, sampler=None, batch_size=batch_size)
        
    n_batches = len(train_loader)
    
    weight = torch.FloatTensor([1.0,95.0]).to(device)
    
    #Time for printing
    training_start_time = time.time()

    train_plot = []
    valid_plot = []
    tss_plot = []
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        decay_rate = learning_rate
        #decay_rate = learning_rate*(1 - epoch / n_epochs)
        #Create our loss and optimizer functions
        loss, optimizer = createLossAndOptimizer(net, decay_rate, weight)
    
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()
            
            #Print statistics
            running_loss += loss_size.data.item()
            total_train_loss += loss_size.data.item()
            
            #Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))


                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        confusion_matrix = torch.zeros(2, 2)
        correct = 0
        total = 0

        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
    
            #Forward pass
            val_outputs = net(inputs)
            _, predicted = torch.max(val_outputs.data, 1)        
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.data.item()
            
        tp = confusion_matrix[1,1]
        tn = confusion_matrix[0,0]
        fp = confusion_matrix[0,1]
        fn = confusion_matrix[1,0]
        
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("TSS = {:.2f}".format((tp) / (tp + fn) - (fp) / (fp + tn)))
        print("TP = {}, FP = {}, FN = {} TN  = {}".format(tp, fp, fn, tn))
        print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

        train_plot.append(total_train_loss / len(train_loader))
        valid_plot.append(total_val_loss / len(val_loader))
        tss_plot.append((tp) / (tp + fn) - (fp) / (fp + tn))
    
    train_plot = np.array(train_plot).reshape((-1,1))
    valid_plot = np.array(valid_plot).reshape((-1,1))
    tss_plot = np.array(tss_plot).reshape((-1,1))
    
    plt.figure()
    plt.plot(train_plot, label = 'train')
    plt.plot(valid_plot, label = 'valid')
    plt.plot(tss_plot, label = 'tss')
    plt.legend()
    plt.savefig('errors.png')
    
    np.save('error', np.concatenate((train_plot, valid_plot, tss_plot), axis = 1)) 

# In[107]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NN = SimpleDNN(in_shape=len(cols));
NN.to(device);

# In[108]:


trainNet(NN, batch_size=128, n_epochs=300, learning_rate=1e-3)
torch.save(NN.state_dict(), './model')


# In[ ]:





# In[ ]:


print(len(sunspotTrainSet.label_frame[(sunspotTrainSet.label_frame == 1)]) / len(sunspotTrainSet.label_frame))


# In[ ]:




