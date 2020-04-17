#!/bin/python
    
import pandas as pd
import numpy as np

#df1 = pd.read_csv('./results/370K/combined.csv')

df1 = pd.read_csv('./current.csv')
df2 = pd.read_csv('~/solar-flares/datasets/all_labels_valid.csv')

df1['merge_label'] = df1['label'].str.replace('(.*).Br.fits', lambda f: f.group(1) + '.png')
#df1['merge_label'] = df1['label'] 
df2['merge_label'] = df2['filename'].str.replace('/srv/data/varad/data/all_images/', '')

print(df2.head())
existing_rows = ~df2['merge_label'].isin(df1['merge_label'].tolist())
df2 = df2.drop(columns=['merge_label'])
print(df2.head())
df2[existing_rows].to_csv('missing.csv', index=None)
np.savetxt('existing.txt', existing_rows)
