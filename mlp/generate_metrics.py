import sys, os
import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt

rootdir = sys.argv[1]
label = sys.argv[2]
short_label = sys.argv[3]

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TSS', 'HSS']

def get_metrics(rootdir, features):

    data = []

    for dir in os.listdir(rootdir):
        results = np.load(rootdir + '/' + dir + '/' + features + '_stats.npy')
        if results.shape[1] != 7:
            continue
        tp = results[-1, 3] 
        fp = results[-1, 4] 
        tn = results[-1, 5] 
        fn = results[-1, 6] 
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        tss = tp / (tp + fn) - fp / (fp + tn)
        hss = 2*(tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        
        data.append(np.array([accuracy, precision, recall, f1, tss, hss]))

        #print(dir, accuracy, precision, recall, f1, tss, hss)

    return np.array(data)

trad = get_metrics(rootdir, 'traditional')
top = get_metrics(rootdir, 'topological')
all = get_metrics(rootdir, 'all')

for i in range(len(metric_names)):
    
    data = np.zeros((trad.shape[0], 5))
    data[:,0] = trad[:,i]
    data[:,1] = top[:,i]
    data[:,2] = all[:,i]
    data[:,3] = top[:,i] - trad[:,i]
    data[:,4] = all[:,i] - trad[:,i]

    print(metric_names[i], data.shape)

    df = pd.DataFrame(data, columns=['SHARPs', label, 'SHARPs_' + short_label, label + ' - SHARPs', 'SHARPs_' + short_label + ' - SHARPs'])

    plt.figure(dpi=300, figsize=(6.8, 6.8))
    plt.title(metric_names[i], fontsize=16)
    df.boxplot(column=['SHARPs', label, 'SHARPs_' + short_label], showmeans=True, meanline=True, fontsize=16)
    plt.ylim([0.5,1])
    plt.xticks(rotation=10)
    fig = plt.gcf()
    size = fig.get_size_inches()
    print(size)
    plt.savefig(label + '_' + metric_names[i] + '.png')

    plt.figure(dpi=300, figsize=(6.8, 6.8))
    plt.title(metric_names[i] + ': Improvement Over Baseline', fontsize=16)
    df.boxplot(column=[label + ' - SHARPs', 'SHARPs_' + short_label + ' - SHARPs'], showmeans=True, meanline=True, fontsize=16)
    plt.ylim([-0.15,0.15])
    plt.xticks(rotation=10)
    plt.savefig(label + '_' + metric_names[i] + '_Improvement.png')

    print(df.mean())
