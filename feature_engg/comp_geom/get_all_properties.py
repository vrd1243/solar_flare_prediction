from threading import Thread
#from queue import Queue, Empty
#from time import sleep
#from urllib.request import urlopen
#from urllib.parse import urlsplit
#from datetime import timedelta
#from collections import defaultdict
from extract_hmi_properties import extract_hmi_properties
from flare_prediction_data import load_sharp_data, get_segment, get_segment_filepath
import argparse
#import logging
import os, sys
import csv
#import concurrent.futures
from joblib import Parallel, delayed
from utilities import line_count_in_file
import multiprocessing

#from drms import Client
import pandas as pd
import numpy as np
from astropy.io import fits

from sklearn.feature_selection import SelectKBest, f_classif

import random
import re

def run_thread(all_data, slot): 
    
    root_dir = '/srv/data/varad/data/fits/'
    fname = 'results_%s_%d.csv' % (flare_label, slot)
    count = line_count_in_file(fname);
    print(count, flare_label);
    
    if os.path.exists(fname):
        df = pd.read_csv(fname, header='infer')
        count = df.shape[0]
        outfile = open(fname, 'a')
    else:
        outfile = open(fname, 'w');

    writer = csv.writer(outfile);

    if not os.path.exists(fname):
        writer.writerow(header + meta_header + ['flare']);
    
    rcomp = re.compile('hmi\.sharp_cea_720s\.(\d+)\.(\d\d\d\d).*')
    
    print(all_data.shape, count)
    for i, row in all_data[count:].iterrows():
        m = re.search(rcomp, row['filename'])
        harpnum = m.group(1)
        hdu = fits.open(root_dir + '/' + 'sharp_' + harpnum + '/' + row['filename'])
        hdu[1].verify('fix')
        segment = hdu[1].data[::4, ::4]
        geom_properties = extract_hmi_properties(segment, row['filename'], do_plot=False); 
        
        for r in geom_properties:
            r.insert(0, i)
            for prop in meta_header:
                r.append(hdu[1].header[prop])
            r.append(row[flare_label]);
        
        writer.writerows(geom_properties);

    outfile.close();

if __name__ == "__main__": 
    
    flare_label = 'any_flare_in_24h'
   # data = pd.read_csv('~/.flare_prediction_data/flare_history_data.csv', parse_dates=[2]);
    data = pd.read_csv('/srv/data/varad/data/fits_labels.csv')
    data.loc[:, 'any_flare_in_24h'] = data.loc[:,'X_flare_in_24h'] + data.loc[:, 'M_flare_in_24h']
    rows_with_2 = np.where(data.loc[:, 'any_flare_in_24h'] == 2)[0]
    print(rows_with_2)
    data.loc[rows_with_2, 'any_flare_in_24h'] = 1

    print(data.head())
    any_flare_in_24h = data[flare_label];
    flare_rows = np.where(any_flare_in_24h == 1)[0];
    no_flare_rows = np.where(any_flare_in_24h == 0)[0];

    # This is only to avoid rework in case of failures. You basically can restart from 
    # the point you crashed.

    print("Total flares ", len(flare_rows), "Total non-flares ", len(no_flare_rows));    

    flare_data = data.loc[flare_rows, :];
    no_flare_data = data.loc[no_flare_rows, :];
    
    all_rows = np.concatenate((flare_rows, no_flare_rows), axis=0);
    all_data = data.loc[all_rows, :];
    #all_data = flare_data

    header = ['row', 'label', 'epsilon', 'total_p', 'total_n', 
              'max_p_size', 'max_n_size', 'max_p_mag', 'max_n_mag', 
              'best_p_size', 'best_n_size', 'best_p_mag', 'best_n_mag', 
              'best_p_density', 'best_n_density', 'com_dist', 'min_dist', 'dist_ratio', 'force']
    
    meta_header = ['LAT_FWT','LON_FWT','AREA_ACR','USFLUX','MEANGAM','MEANGBT','MEANGBZ','MEANGBH','MEANJZD','TOTUSJZ','MEANALP','MEANJZH','TOTUSJH','ABSNJZH','SAVNCPP','MEANPOT','TOTPOT','MEANSHR','SHRGT45']
    #num_cores = multiprocessing.cpu_count() - 1;
    num_cores = 8
    slots = np.arange(0, all_data.shape[0], int(all_data.shape[0]/num_cores));   
    print(all_data.shape);
    
    Parallel(n_jobs=num_cores)(delayed(run_thread)(all_data.iloc[slots[i]:slots[i+1]], i) for i in np.arange(slots.shape[0] - 1))

