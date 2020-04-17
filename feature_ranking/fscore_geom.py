#!/bin/python

import numpy as np
import pandas as pd
from sklearn import feature_selection
from matplotlib import pyplot as plt
from matplotlib import rcParams

df = pd.read_csv('./geometry.csv')

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

X = df.iloc[:,2:-1]
y = df.iloc[:,-1]

columns = X.columns
print(columns)

F,pval = feature_selection.f_classif(X,y)

F_idx = np.argsort(F)[::-1]

for idx in F_idx:
    print(columns[idx], F[idx] / F[F_idx[0]])

colors = ['red' if idx < 16 else 'blue' for idx in range(len(columns))] #range(1,df.shape[1] - 1)]
labels = ['Geometry' if idx < 16 else 'SHARP' for idx in range(len(columns))] #range(1,df.shape[1] - 1)]

fig, ax = plt.subplots(figsize=(12,8))
plt.xticks(np.arange(35))
ax.tick_params(axis='both', labelsize=14)

x_geometry = [np.where(F_idx == i)[0][0] for i in range(16)]
y_geometry = [F[F_idx[np.where(F_idx == i)[0][0]]] / F[F_idx[0]] for i in range(16)]

geometry_idx = np.where(np.array(x_geometry) < 15)[0]

print("Geometry")
print(x_geometry, y_geometry)

x_hmi = [np.where(F_idx == i)[0][0] for i in range(16,35)]
y_hmi = [F[F_idx[np.where(F_idx == i)[0][0]]] / F[F_idx[0]] for i in range(16,35)]

hmi_idx = np.where(np.array(x_hmi) < 15)[0]

print(x_hmi, y_hmi)
#print(columns)

plt.figure(dpi=300)
top = plt.scatter(np.array(y_geometry)[geometry_idx], np.array(x_geometry)[geometry_idx] + 1,  color='red', s = 100)
hmi = plt.scatter(np.array(y_hmi)[hmi_idx], np.array(x_hmi)[hmi_idx] + 1, color='blue', s = 100)

plt.legend((top, hmi), ('Geometry', 'SHARP'), fontsize=14)

for plt_idx, idx in enumerate(F_idx[:15]):
    x = plt_idx
    y = F[idx] / F[F_idx[0]]
    plt.text(y-0.05,x+1, columns[idx] ,fontsize=9, rotation=0, ha='left', va='center', color=colors[idx], family = 'monospace')

plt.xlim([1.1,-.2])
plt.ylim([16,-1])
plt.xlabel('Normalized F-Score Value', fontsize=14)
plt.ylabel('Feature Rank', fontsize=14)
plt.savefig('feature_ranking_geometry.png')

for i in range(len(colors)):
    print(i, columns[i], colors[i])
