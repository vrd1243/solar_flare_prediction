#!/bin/python

import torch
import re
import random

# Generate a torch.utils.data.Dataset that can be used by
# the dataLoaderClass and be batched and shuffled during
# training.

class sunspotPropertiesDataSet(torch.utils.data.Dataset):

    def __init__(self, df, cols, transform):
        self.prop_frame = df.iloc[:, cols].astype('float32')
        self.label_frame = df.iloc[:, -1].astype('long')
        self.transform = transform

    def __len__(self):

        return len(self.prop_frame)

    def __getitem__(self, idx):

        properties = self.prop_frame.iloc[idx, :].values
        labels = self.label_frame.iloc[idx]

        return [properties, labels]

def generateTrainValidData(df, cols, splitType = 'random'):
    
    if splitType == 'random':
        df_train = df.sample(frac=0.7, random_state=1)
        print(df_train.head())
    
    elif splitType == 'temporal':
        pattern = re.compile('hmi.sharp_cea_720s\..*\.(\d\d\d\d).*')
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['label']).group(1)) <= 2014, axis=1)]
    
    elif splitType == 'by_harpnum':   
        pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
        harpnum = df['label'].str.extract(pattern).astype('int64')
        harpnum_set = harpnum[0].unique()
        random.seed(1000)
        random.shuffle(harpnum_set)
        split = int(0.7*harpnum_set.shape[0]) 
        train_harpnums = harpnum_set[:split]
        print(train_harpnums)
        #test_harpnums = harpnum_set[split]
        df_train = df[df.apply(lambda x: int(re.search(pattern, x['label']).group(1)) in train_harpnums, axis=1)]

    df_valid = df.loc[~df.index.isin(df_train.index)]

    print("These are the columns ", df_valid.columns[cols])

    trainDataSet = sunspotPropertiesDataSet(df_train, cols=cols, transform=None)
    validDataSet = sunspotPropertiesDataSet(df_valid, cols=cols, transform=None)
    return [trainDataSet, validDataSet]
