import sys, os
import numpy as np
import glob
import pandas as pd
from matplotlib import pyplot as plt

rootdir = sys.argv[1]

metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'TSS', 'HSS']

def get_metrics(rootdir):

    data = []

    for dir in os.listdir(rootdir):
        results = np.load(rootdir + '/' + dir + '/' + 'stats.npy')
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

data = get_metrics(rootdir)
df = pd.DataFrame(data, columns=metric_names)
#trad = get_metrics(rootdir, 'traditional')
#top = get_metrics(rootdir, 'topological')
#all = get_metrics(rootdir, 'all')


plt.figure(dpi=300, figsize=(10,6))
plt.title('CNN', fontsize=16)
df.boxplot(column=metric_names, showmeans=True, meanline=True, fontsize=16)
plt.savefig('stats.png')
