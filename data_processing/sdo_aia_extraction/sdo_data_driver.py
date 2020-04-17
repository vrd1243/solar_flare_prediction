#!/bin/python

from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
import os
#import example_helpers
import drms
import sys
from get_flare_data import load_flare_catalogue, noaa_to_harp_data
from datetime import timedelta
import csv
from extract_sdo import extract_hmi_sdo

start_time = sys.argv[3];
#tsel = '2010.01.01_00:00:00_TAI/3400d@12m'
tsel = start_time + '/4000d@12m'

series = 'hmi.sharp_720s'
export_protocol = 'fits'
segments = ['magnetogram']

flares = load_flare_catalogue();
flares.to_csv('/srv/data/varad/data/sdo_dataset_flares.csv', sep='\t')
harp_to_noaa_dict = noaa_to_harp_data();

tds = [timedelta(hours=1), timedelta(hours=3), timedelta(hours=6), timedelta(hours=12), timedelta(hours=24), timedelta(hours=48)]
kind = ['M', 'X']
columns = ['timestamp']

for td in tds:
    hours = td.total_seconds() / 3600
    for k in kind:
        columns.append("%s_flare_in_%dh" % (k, hours))

labels_file = open("/srv/data/varad/data/sdo_dataset_labels.csv", 'a')
writer = csv.writer(labels_file, delimiter = ',')
writer.writerow(columns)

for harpnum in range(int(sys.argv[1]),int(sys.argv[2])):
    print("Processing ", harpnum)
#try: 
    c = drms.Client(verbose=True)
    qstr = '%s[%d][%s]' % (series, harpnum, tsel)
    print("Querying timestamps ... ");
    r, segments = c.query(qstr, key='T_REC', seg = 'magnetogram')
    print("Timestamp query completed ", r.T_REC.shape);
    
    for idx in np.arange(len(r.T_REC[:-1])):
        extract_hmi_sdo(harpnum, r.T_REC[idx], r.T_REC[idx+1], harp_to_noaa_dict, flares, writer, tds);
#except:
#    continue 
