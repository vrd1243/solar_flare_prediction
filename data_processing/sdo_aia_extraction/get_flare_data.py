
from threading import Thread
from queue import Queue, Empty
from time import sleep
from urllib.request import urlopen
from urllib.parse import urlsplit
from datetime import timedelta
from collections import defaultdict
import argparse
import logging
import os

from drms import Client
import pandas as pd
import numpy as np
from astropy.io import fits

harp_to_noaa_url = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"
flare_catalogue_url = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/" \
                      "solar-flares/x-rays/goes/xrs/goes-xrs-report_%d.txt"

data_dir = os.path.expanduser("~/.flare_prediction_data")

def _cacheopen(url, file_path=None):
    """
    Return a file object of a resource from the local cache,
    downloading it is not present in the cache.
    :param url: url of the resource
    :param file_path:
    :return:
    """
    if file_path is None:
        file_path = os.path.split(urlsplit(url).path)[1]
        file_path = os.path.join(data_dir, file_path)
    if not os.path.exists(file_path):
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        with urlopen(url) as infile, open(file_path, "wb") as outfile:
            outfile.write(infile.read())
    return open(file_path)

def load_flare_catalogue():
    """Load the flare catalogue, and return it in a data frame."""
    urls = [flare_catalogue_url % i for i in range(1975, 2017)]
    data = []

    for url in urls:
        with _cacheopen(url) as f:
            for line in f:
                if len(line) <= 59:
                    continue
                class_ = line[59]
                noaa = line[80:85].strip()
                if noaa and class_ in ["M", "X"]:
                    start = pd.to_datetime(line[5:11] + line[13:17], format="%y%m%d%H%M")
                    end_hour = int(line[18:20])
                    end_minute = int(line[20:22])
                    # The file only specifies the start and end time, and not full date. If the flare ends after
                    # midnight, it doesn't wrap back around to 0, it goes past 23.
                    if end_hour > 23:
                        end = (start + timedelta(days=1)).replace(hour=end_hour % 24, minute=end_minute)
                    else:
                        end = start.replace(hour=end_hour, minute=end_minute)
                    if line[28] in ["N", "S"]:
                        latitude = int(line[29:31]) * (-1 if line[28] == "S" else 1)
                    else:
                        latitude = None
                    data.append((int(noaa), class_, latitude, start, end))

    return pd.DataFrame(data, columns=["noaa", "class", "latitude", "start", "end"])


def noaa_to_harp_data():
    with _cacheopen(harp_to_noaa_url) as data:
        lines = data.read().splitlines()
        harp_to_noaa_dict = {};
        for line in lines[1:]:
            harpnum, noaa = line.split();
            #for n in noaa.split(','):
                #if int(n) in harp_to_noaa_dict:
                    #print(n, harp_to_noaa_dict[int(n)])
                #harp_to_noaa_dict[int(n)] = int(harpnum)
            harp_to_noaa_dict[int(harpnum)] = [int(n) for n in noaa.split(',')]
     
    for i in range(max(harp_to_noaa_dict.keys())):
        if i not in harp_to_noaa_dict.keys():
            harp_to_noaa_dict[i] = [-1]

    return harp_to_noaa_dict
