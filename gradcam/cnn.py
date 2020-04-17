#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torchvision
import torch

from torch.autograd import Variable
import torch.nn.functional as F

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import os
from PIL import Image
import data


# In[2]:


class SimpleCNN(torch.nn.Module):
        
    #Our batch shape for input x is (4, 256, 256)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
       
        # 256 x 256 x 4
        #Input channels = 4, output channels = 8
        self.conv1 = torch.nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 128 x 128 x 8
        #Input channels = 8, output channels = 16
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 64 x 64 x 16
        #Input channels = 16, output channels = 32
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # 32 x 32 x 32
        #Input channels = 32, output channels = 64
        self.conv4 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        #4608 input features, 64 output features (see sizing flow below)
        self.fc1 = torch.nn.Linear(32 * 32 * 32, 256)
        
        #64 input features, 10 output features for our 2 defined classes
        self.fc2 = torch.nn.Linear(256, 64)
        
        self.fc3 = torch.nn.Linear(64, 2)
        
        self.dropout = torch.nn.Dropout(0.01)
        
    def forward(self, x):
        
        #Computes the activation of the first convolution
        #Size changes from (4, 256, 256) to (8, 256, 256)
        x = F.relu(self.conv1(x))
        
        #Size changes from (8, 256, 256) to (8, 128, 128)
        x = self.pool1(x)
        
        #Size changes from (8, 128, 128) to (16, 128, 128)
        x = F.relu(self.conv2(x))
        
        #Size changes from (16, 128, 128) to (16, 64, 64)
        x = self.pool2(x)
        
        #Size changes from (16, 64, 64) to (32, 64, 64)
        x = F.relu(self.conv3(x))
        
        #Size changes from (32, 64, 64) to (32, 32, 32)
        x = self.pool3(x)
        
        #Size changes from (32, 32, 32) to (64, 32, 32)
        #x = F.relu(self.conv4(x))
        
        #Size changes from (64, 32, 32) to (64, 16, 16)
        #x = self.pool4(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (32, 32, 32) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        x = x.view(-1, 32 * 32 * 32)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 32768) to (1, 64)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        
        #x = self.dropout(x)
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 2)
        x = self.fc3(x)
        
        #return(F.log_softmax(x, dim=1))
        return x


# In[3]:


torch.cuda.current_device()
torch.cuda.device(0)
torch.cuda.device_count()
torch.cuda.get_device_name(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[4]:


class sunspotDataset(torch.utils.data.Dataset):

    def __init__(self,text_file,root_dir,transform):
        """
        Args:
            text_file(string): path to text file
            root_dir(string): directory with all train images
        """
        self.name_frame = pd.read_csv(text_file,sep=",",usecols=range(1), header='infer')
        self.label_frame = pd.read_csv(text_file,sep=",",usecols=range(1,2), header='infer')
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx, 0])
        image = Image.open(img_name)
        image = image.resize((256, 256))#, Image.AFFINE)
        image = self.transform(image)
    
        labels = self.label_frame.iloc[idx, 0]
        #labels = labels.reshape(-1, 2)
        #sample = {'image': image, 'labels': labels}

        return [image, labels]


# In[10]:


#sunspotTrainSet = sunspotDataset(df_train, root_dir = '../../data/all_images/', transform=transform)
#sunspotValidSet = sunspotDataset(df_valid, root_dir = '../../data/all_images/', transform=transform)
import importlib
import data
importlib.reload(data)

df = pd.read_csv("~/solar-flares/datasets/all_labels_sorted_small_dataset.csv", sep=",", header='infer')
#df = pd.read_csv("~/solar-flares/datasets/all_labels_debug.csv", sep=",", header='infer')
sunspotTrainSet, sunspotValidSet = data.generateTrainValidData(df, root_dir='/', splitType='by_harpnum')

def get_loader(set, sampler, batch_size):    
    sunspotLoader = torch.utils.data.DataLoader(set, sampler=sampler, num_workers=2, batch_size=batch_size, shuffle=True)
    
    return sunspotLoader


# In[11]:


import torch.optim as optim

def createLossAndOptimizer(net, learning_rate, weight):
        
    reg_loss = 0
    for param in net.parameters():
        reg_loss += torch.sum(torch.abs(param))

    #Loss function
    loss = torch.nn.CrossEntropyLoss(weight=weight)
    
    #Optimizer
    optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, weight_decay=0.001, lr_decay=1e-4)
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    
    return(loss, optimizer)


# In[12]:


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
    
    weight = torch.FloatTensor([1,100]).to(device)
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate, weight)
    
    #Time for printing
    training_start_time = time.time()
    
    statistics = []
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        net.train()
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
        
        net.eval()
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


# In[13]:




# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

CNN = SimpleCNN();
CNN.to(device);

torch.save(CNN.state_dict(), './cnn_random_before_training')

# In[ ]:


trainNet(CNN, batch_size=128, n_epochs=25, learning_rate=1e-3)
torch.save(CNN.state_dict(), './cnn_random_after_training')


# In[15]:


correct = 0
total = 0

val_loader = get_loader(sunspotValidSet, sampler=None, batch_size=64)
confusion_matrix = torch.zeros(2, 2)
with torch.no_grad():
    for data in val_loader:
        #Get inputs
        inputs, labels = data
            
        #Wrap them in a Variable object
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = CNN(inputs)
        
        _, predicted = torch.max(outputs.data, 1)        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for t, p in zip(labels.view(-1), predicted.view(-1)):
            confusion_matrix[t.long(), p.long()] += 1

print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))


# In[18]:


print(confusion_matrix)


# In[17]:


tp = confusion_matrix[1,1]
tn = confusion_matrix[0,0]
fp = confusion_matrix[0,1]
fn = confusion_matrix[1,0]

print((tp) / (tp + fn) - (fp) / (fp + tn))


# In[ ]:




