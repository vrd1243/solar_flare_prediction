import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from flare_prediction_data import load_sharp_data 
import sys

in_file = 'results.csv'

if len(sys.argv) > 1:
    in_file = sys.argv[1];

data = pd.read_csv(in_file);

print(data.shape);
data = data.replace([np.inf, -np.inf], np.nan)
#sharp_data = load_sharp_data();
data = data.dropna(axis = 0);
print(data.shape);

X = data.iloc[:, 3:-1];
y = data.iloc[:, -1];

fit = SelectKBest(f_classif, k = 'all').fit(X, y);
labels = data.columns[3:-1];

best_scores, best_labels = zip(*sorted(zip(fit.scores_, labels), reverse = True));

#print(best_labels);
#print(best_scores);
#print(best_scores/best_scores[0]);

for idx in range(len(best_labels)):
    print(best_labels[idx], best_scores[idx]/best_scores[0])

for idx in range(len(best_labels)):
  plt.figure();
  plt.hist(data.loc[np.where(y == 0)[0], best_labels[idx]], bins=100);
  plt.hist(data.loc[np.where(y == 1)[0], best_labels[idx]], bins=100);
  plt.axvline(x = np.mean(data.loc[np.where(y == 0)[0], best_labels[idx]]));
  plt.axvline(x = np.mean(data.loc[np.where(y == 1)[0], best_labels[idx]]));
  plt.savefig('best_score_dist_%d.png' % (idx));
