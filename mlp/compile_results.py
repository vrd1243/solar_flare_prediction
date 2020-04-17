import sys, os
import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt

rootdir = sys.argv[1]
label = sys.argv[2]

data = []

for dir in os.listdir(rootdir):
    print(dir)
    trad = np.load('results/common/geometry/' + dir + '/' + 'traditional_stats.npy')[-1,2]
    top = np.load(rootdir + '/' + dir + '/' + 'topological_stats.npy')[-1,2]
    all = np.load(rootdir + '/' + dir + '/' + 'all_stats.npy')[-1,2]

    data.append(np.array([trad, top, all, top - trad, all - trad]))

df = pd.DataFrame(np.array(data), columns=['SHARP', label, 'Both', label + ' - SHARP', 'Both - SHARP'])

print(df.mean())
print(df.min())
print(df.max())
print(df)

plt.figure(dpi=300)
plt.title('TSS: ' + label, fontsize=16)
df.boxplot(column=['SHARP', label, 'Both'], showmeans=True, meanline=True, fontsize=16)
plt.savefig(label + '.png')

plt.figure(dpi=300)
plt.title('TSS Difference: ' + label, fontsize=16)
df.boxplot(column=[label + ' - SHARP', 'Both - SHARP'], showmeans=True, meanline=True, fontsize=16)
plt.savefig(label + '_Difference.png')
