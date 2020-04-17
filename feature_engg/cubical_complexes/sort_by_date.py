#!/bin/python

import numpy as np
import pandas as pd
import re
import datetime

df = pd.read_csv('./results/merged.csv', header='infer')

#demo_pattern = re.compile('hmi.sharp_cea_720s\..*\.(?P<year>\d\d\d\d)(?P<month>\d\d)(?P<day>\d\d)_(?P<hour>\d\d)(?P<min>\d\d)(?P<sec>\d\d).*')
#date_df = df['filename'].str.extract(demo_pattern)

date_pattern = 'hmi.sharp_cea_720s\..*\.(\d\d\d\d\d\d\d\d_\d\d\d\d\d\d).*'
#df.apply(lambda x: re.search(pattern, x['filename']).group(1), axis=1)
harpnum_pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
df['date'] = df.apply(lambda x: datetime.datetime.strptime(re.search(date_pattern, x['0']).group(1), '%Y%m%d_%H%M%S'), axis=1)
df['harpnum'] = df.apply(lambda x: re.search(harpnum_pattern, x['0']).group(1), axis=1).astype('int64')
df = df.sort_values(by = ['harpnum', 'date'], ascending = [True,True])#.drop(['harpnum', 'date'], axis=1)
print(df.columns)
df.to_csv('./results/merged_sorted.csv', index=False)

print(df.head())
#print(date_df.head())

