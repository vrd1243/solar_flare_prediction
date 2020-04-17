import astropy
from astropy.io import fits
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, shutil, glob
import networkx as nx
import sys
import matplotlib.patches as patches
import subprocess
from subprocess import call
from homology import *
import pandas as pd
from flare_prediction_data import load_sharp_data, get_segment, get_segment_filepath
from prf_dists import *
from itertools import groupby

def makedirs(path, suffix = 'out'):
  
  path = './results/' + path + '_' + suffix;
  try:
      os.stat(path)
      shutil.rmtree(path);
      os.mkdir(path) 

  except:
      os.mkdir(path) 

  return path

def compute_toplexes(data, label, do_plot = False):
        
    path = makedirs(label);
    caller_dir = os.getcwd();
    os.chdir(path);
    
    if do_plot:

        plt.figure();
        plt.matshow(data - np.min(data));
        plt.savefig(label + '_in.png')

    b0_intervals = Intervals(data, '0');
    b0_pd = PD(b0_intervals);
    b0_prf = PRF(b0_pd);
    np.save(label + '_b0', b0_prf);
    
    b1_intervals = Intervals(data, '1');
    b1_pd = PD(b1_intervals);
    b1_prf = PRF(b1_pd);
    np.save(label + '_b1', b1_prf);

    os.chdir(caller_dir);
    #b0_arr[b0_arr == -1] = np.max(b0_arr[:,1])
    #b1_arr[b1_arr == -1] = np.max(b1_arr[:,1])
    return [b0_prf, b1_prf];


def prf_dists_for_harp(harpnum, harp_data, first_row):    
    
     
    if os.path.exists('results/' + str(harpnum) + '_' + str(first_row) + '.txt'):
        return;

    b0_prf_pos = [];
    b1_prf_pos = [];
    b0_prf_neg = [];
    b1_prf_neg = [];
    
    invalid_indices = [];
    print("First row before processing ", first_row);
        
    for i, row in harp_data.iterrows():
        if not os.path.exists(get_segment_filepath('magnetogram', '../flare_prediction_v0.1/data', row['T_REC'], row['HARPNUM'], 'fits')):
            print("Segment not found .. Skipping");
            b0_prf_pos.append(-1);
            b1_prf_pos.append(-1);
            b0_prf_neg.append(-1);
            b1_prf_neg.append(-1);
            continue;

        segment = get_segment('magnetogram', '../flare_prediction_v0.1/data', row['T_REC'], row['HARPNUM'], 'fits');
        data = segment[::4, ::4]

        if np.isnan(data).any():
            print("Data contains nan. Adding to invalid data indices");
            b0_prf_pos.append(-1);
            b1_prf_pos.append(-1);
            b0_prf_neg.append(-1);
            b1_prf_neg.append(-1);
            continue;

        pos_data = np.copy(data);
        pos_data[data < 0] = 0;
        neg_data = np.copy(data);
        neg_data[data > 0] = 0;
        neg_data = -neg_data;
        
        [b0, b1] = compute_toplexes(pos_data, str(row['HARPNUM']) + '_' + str(row['T_REC']) + '_pos');
        
        b0_prf_pos.append(b0);
        b1_prf_pos.append(b1);

        [b0, b1] = compute_toplexes(neg_data, str(row['HARPNUM']) + '_' + str(row['T_REC']) + '_neg');

        b0_prf_neg.append(b0);
        b1_prf_neg.append(b1); 
	
    b0_prf_dists_pos = consecutive_dists(b0_prf_pos);
    b1_prf_dists_pos = consecutive_dists(b1_prf_pos);
    b0_prf_dists_neg = consecutive_dists(b0_prf_neg);
    b1_prf_dists_neg = consecutive_dists(b1_prf_neg);
    
    #print(harp_data['HARPNUM'].shape);
    #np.savetxt('results/' + str(harpnum) + '_' + str(first_row) + '.txt', 
    #           np.concatenate((np.array(harp_data['ROW']).reshape((-1,1)), 
    #	                       b0_prf_dists_pos.reshape((-1,1)), b1_prf_dists_pos.reshape((-1,1)), 
    #                           b0_prf_dists_neg.reshape((-1,1)), b1_prf_dists_neg.reshape((-1,1))),
    #			       axis = 1));
    
    plt.figure();
    plt.plot(b0_prf_dists_pos);
    plt.plot(b1_prf_dists_pos);
    plt.plot(b0_prf_dists_neg);
    plt.plot(b1_prf_dists_neg);
    plt.savefig('results/' + str(harpnum) + '_' + str(first_row) + '.png');

if __name__ == '__main__':

    data = pd.read_csv('~/.flare_prediction_data/flare_history_data.csv', parse_dates=[2]);
    
    harpnums = set(data['HARPNUM']);
    grouped = [(k, sum(1 for i in g)) for k,g in groupby(data['HARPNUM'])]
    
    start = 0; 
    for g in grouped:
        harpnum = g[0];
        end = g[1] + start; 
        print(g[0], g[1]);
        prf_dists_for_harp(harpnum, data[start:end], start);
        start = end + 1;

        
