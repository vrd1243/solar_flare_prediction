#!/bin/python

import torch
import re
import random
import os
import numpy as np
from PIL import Image
from torchvision import transforms, utils

# Inherit from the torch.utils.data.Dataset, so that this can be 
# used by DataLoader class, and split into batches for training/validation.
# This dataset loads the Br component of vector magnetogram.
class sunspotImageDataSetBr(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform):

        self.name_frame = df.iloc[:, 0]
        self.label_frame = df.loc[:, 'any_flare_in_24h']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(self.name_frame)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx])
        image = Image.open(img_name)
        image = image.resize((256, 256), Image.AFFINE)
        image = self.transform(image)
        image = np.array(image)[1,:,:].reshape((1, image.shape[1], image.shape[2]))
        labels = self.label_frame.iloc[idx]

        return [image, labels]

# Inherit from the torch.utils.data.Dataset, so that this can be 
# used by DataLoader class, and split into batches for training/validation.
# This dataset loads the full vector magnetogram.
class sunspotImageDataSet(torch.utils.data.Dataset):

    def __init__(self, df, root_dir, transform):

        self.name_frame = df.iloc[:, 0]
        self.label_frame = df.loc[:, 'any_flare_in_24h']
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):

        return len(self.name_frame)

    def __getitem__(self, idx):
        
        img_name = os.path.join(self.root_dir, self.name_frame.iloc[idx])
        image = Image.open(img_name)
        image = image.resize((256, 256), Image.AFFINE)
        image = self.transform(image)
        labels = self.label_frame.iloc[idx]

        return [image, labels]

# Create datasets from the file containing image name and labels.
def generateTrainValidData(df, root_dir, splitType = 'by_harpnum'):
    
    df['any_flare_in_24h'] = df['M_flare_in_24h'] + df['X_flare_in_24h']
    rows = df.loc[:,'any_flare_in_24h'] == 2
    df.loc[rows, 'any_flare_in_24h'] = 1

    if splitType == 'random':
        df_train = df.sample(frac=0.7, random_state=1)
    
    elif splitType == 'temporal':
        pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['label']).group(1)) <= 2014, axis=1)]
    
    elif splitType == 'by_harpnum':   
        pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
        harpnum = df['filename'].str.extract(pattern).astype('int64')
        harpnum_set = harpnum[0].unique()
        random.seed(1000)
        random.shuffle(harpnum_set)
        split = int(0.7*harpnum_set.shape[0]) 
        train_harpnums = harpnum_set[:split]
        print(train_harpnums)
        #test_harpnums = harpnum_set[split]
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['filename']).group(1)) in train_harpnums, axis=1)] 

    df_valid = df.loc[~df.index.isin(df_train.index)]

    dataSetFn = sunspotImageDataSet
    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5,0.5), (0.5,0.5,0.5,0.5))])
    trainDataSet = dataSetFn(df_train, root_dir, transform=transform)
    validDataSet = dataSetFn(df_valid, root_dir, transform=transform)
    return [trainDataSet, validDataSet]
