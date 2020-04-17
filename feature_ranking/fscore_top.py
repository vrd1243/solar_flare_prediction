#!/bin/python

import numpy as np
import pandas as pd
from sklearn import feature_selection
from matplotlib import pyplot as plt

df = pd.read_csv('topology.csv')

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X = df.iloc[:,list(range(1,20)) + list(range(30,40)) + list(range(50,60))]
y = df.iloc[:,-1]

columns = X.columns
print(columns)

F,pval = feature_selection.f_classif(X,y)

for i in range(len(F)):
    if np.isnan(F[i]):
        F[i] = 0

F_idx = np.argsort(F)[::-1]

for idx in F_idx:
    print(columns[idx], F[idx] / F[F_idx[0]])

colors = ['blue' if idx < 19 else 'red' for idx in range(len(columns))] #range(1,df.shape[1] - 1)]
labels = ['SHARP' if idx < 19 else 'Topology' for idx in range(len(columns))] #range(1,df.shape[1] - 1)]

fig, ax = plt.subplots(figsize=(12,8))
plt.xticks(np.arange(35))
ax.tick_params(axis='both', labelsize=14)

x_topological = [np.where(F_idx == i)[0][0] for i in range(19,39)]
y_topological = [F[F_idx[np.where(F_idx == i)[0][0]]] / F[F_idx[0]] for i in range(19,39)]

topological_idx = np.where(np.array(x_topological) < 15)[0]

print("Topological")
print(x_topological, y_topological)

x_hmi = [np.where(F_idx == i)[0][0] for i in range(19)]
y_hmi = [F[F_idx[np.where(F_idx == i)[0][0]]] / F[F_idx[0]] for i in range(19)]

hmi_idx = np.where(np.array(x_hmi) < 15)[0]


print(x_hmi, y_hmi)

plt.figure(dpi=300)
top = plt.scatter(np.array(y_topological)[topological_idx], np.array(x_topological)[topological_idx] + 1,  color='red', s = 100)
hmi = plt.scatter(np.array(y_hmi)[hmi_idx], np.array(x_hmi)[hmi_idx] + 1, color='blue', s = 100)

plt.legend((top, hmi), ('Topology', 'SHARP'), fontsize=14)

for plt_idx, idx in enumerate(F_idx[:15]):
    x = plt_idx
    y = F[idx] / F[F_idx[0]]
    #plt.scatter(x, y, color=colors[idx]) 
    #plt.text(x+.1, y+.3, columns[idx] ,fontsize=9, rotation=45, ha='left', va='top')
    plt.text(y-0.05,x+1, columns[idx] ,fontsize=9, rotation=0, ha='left', va='center', color=colors[idx], family = 'monospace')
    print(columns[idx], colors[idx])
plt.xlim([1.1,-.2])
plt.ylim([16,-1])
plt.xlabel('Normalized F-Score Value', fontsize=14)
plt.ylabel('Feature Rank', fontsize=14)
plt.savefig('feature_ranking_topology.png')

for i in range(len(colors)):
    print(i, columns[i], colors[i])
