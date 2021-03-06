#!/bin/python

import pandas as pd
import numpy as np
import csv
from process_magnetograms import compute_toplexes
from homology import *
from joblib import Parallel, delayed
import re
from PIL import Image

def run_thread(harp_df, root_dir, slot, ref_df):
    
    b0_prf_pos = []
    b0_prf_neg = []
    b1_prf_pos = []
    b1_prf_neg = []

    fname = 'results_%d.csv' % (slot)
    
    count = 0
    if os.path.exists(fname):
        df = pd.read_csv(fname, header='infer')
        count = df.shape[0]
        outfile = open(fname, 'a')
    else:
        outfile = open(fname, 'w');

    writer = csv.writer(outfile);

    if not os.path.exists(fname):
        header = ['filepath']

        for i in range(50):
            header += ['b1_pos_' + str(i)]

        for i in range(50):
            header += ['b1_neg_' + str(i)]
        
        header += harp_df.columns[1:]
        print(header)
        writer.writerow(header);
    
    print(count,harp_df.shape) 
    harp_df = harp_df.iloc[count:, :]
    print(harp_df.shape) 
    pattern = re.compile('hmi.sharp_cea_720s\.(\d+)\..*')
    valid_rows = []
    
    epsilons = np.linspace(0, 255, 20)
    
    for i, row in harp_df.iterrows():
        if row['valid'] == 0:
            continue

        if ref_df[ref_df.iloc[:,0] == row['filename']].shape[0] != 0:
            continue

        filepath = root_dir + '/' + row['filename']
        image = Image.open(filepath)

        data = np.array(image)
        data = data[:,:,1]
        
        if np.isnan(data).any():
            continue
        
        valid_rows.append(row)
        data_pos = np.copy(data)
        data_pos[data_pos <= 127] = 127
        data_neg = 255 - data
        data_neg[data_neg <= 127] = 127
        
        try:
            [b1_pos] = compute_toplexes(data_pos, epsilons);
            b1_prf_pos.append(b1_pos.data);

            [b1_neg] = compute_toplexes(data_neg, epsilons);
            b1_prf_neg.append(b1_neg.data);

            write_row = [row['filename']] + list(b1_pos) + list(b1_neg) + list(row[1:])
            print(write_row)

            writer.writerow(write_row)

        except:
            print("Couldnt process ", row)
            continue


    prfs = np.concatenate((b1_prf_pos, b1_prf_neg), axis=0)
    np.save('prfs', prfs)
    #return [valid_rows, prfs]
    return



if __name__ == '__main__':
    
    data = pd.read_csv('/srv/data/varad/data/all_labels_debug.csv', header='infer')
    ref_data = pd.read_csv('./results/cubical_complexes_320K_debug.csv', header='infer')
    valid = pd.read_csv('/srv/data/varad/data/valid_entries.csv', header='infer')
    data = pd.merge(data, valid, on='filename')

    num_cores = 16
    slots = np.arange(0, data.shape[0], int(data.shape[0]/num_cores));   
    #[valid_rows, prfs] = create_prfs(data.iloc[:100,:], '/');
    Parallel(n_jobs=num_cores)(delayed(run_thread)(data.iloc[slots[i]:slots[i+1]], '/', i, ref_data) for i in np.arange(slots.shape[0] - 1))

