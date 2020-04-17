
from threading import Thread
from queue import Queue, Empty
from time import sleep
from urllib.request import urlopen
from urllib.parse import urlsplit
from datetime import timedelta
from collections import defaultdict
from extract_hmi_properties import extract_hmi_properties
import argparse
import logging
import os
import csv
import concurrent.futures

from drms import Client
import pandas as pd
import numpy as np
from astropy.io import fits


flare_catalogue_url = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/" \
                          "solar-flares/x-rays/goes/xrs/goes-xrs-report_%d.txt"
sharp_data_url = "https://www.dropbox.com/s/boyk2hpcw1bv1xt/data.csv?dl=1"

data_dir = os.path.expanduser("~/.flare_prediction_data")
output_path = os.path.join(data_dir, "data.csv")
output_path_with_labels = os.path.join(data_dir, "data_with_labels.csv")
failed_path = os.path.join(data_dir, "failed_%s.txt")

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

def _assign_labels(data):
    """
    Label SHARP data and write a local copy.
    :param data: data frame of the SHARP data.
    """
    flares = load_flare_catalogue()
    tds = [timedelta(hours=6), timedelta(hours=12), timedelta(hours=24), timedelta(hours=48)]
    labels = defaultdict(list)
    dfl = len(data)
    for i, row in data.iterrows():
        t0 = row["T_REC"]
        noaa = row["NOAA_AR"]
        for td in tds:
            t1 = t0 + td
            df_td = flares[flares["start"].between(t0, t1)]
            df_td = df_td[df_td["noaa"] == noaa]
            x_flare_occured = int(len(df_td[df_td["class"] == "X"]) != 0)
            m_flare_occured = int(len(df_td[df_td["class"] == "M"]) != 0)
            any_flare_occured = x_flare_occured or m_flare_occured
            labels[td, "X"].append(x_flare_occured)
            labels[td, "M"].append(m_flare_occured)
            labels[td, "any"].append(any_flare_occured)
        if i % 1000 == 0:
            logging.info("Labeled %d/%d" % (i, dfl))

    for (td, kind), col in labels.items():
        hours = td.total_seconds() / 3600
        data["%s_flare_in_%dh" % (kind, hours)] = col

    data.to_csv(output_path_with_labels, index=False)

def get_segment_filepath(seg, basedir, t_rec, harpnum, ext):
    """
    Return a filepath to a segment of the filesystem.
    :param seg: string of which segment
    :param basedir: filepath to the base directory
    :param t_rec: datetime of the obeservation
    :param harpnum: harpnum of the observation
    :param ext: extension of the file
    :return: filepath to the segment
    """
    return os.path.join(basedir, seg, t_rec.strftime("%Y/%m/%d"),
                        t_rec.strftime("%%H-%%M-%%S--%d.%s" % (harpnum, ext)))

def get_segment(seg, basedir, t_rec, harpnum, typ):
    """
    Return a segment from the filesystem.
    :param seg: string of which segment
    :param basedir: filepath to the base directory
    :param t_rec: datetime of the observation
    :param harpnum: harpnum of the observation
    :param typ: type of file to load as; either fits or npy
    :return: numpy array of the segment
    """
    if typ == "fits":
        return fits.getdata(get_segment_filepath(seg, basedir, t_rec, harpnum, ext="fits"))
    elif typ == "npy":
        return np.load(get_segment_filepath(seg, basedir, t_rec, harpnum, ext="npy"))
    else:
        raise ValueError("typ argument must be 'fits' or 'npy'")

def load_sharp_data():
    """Load the SHARP data, and return it in a data frame."""
    return pd.read_csv(_cacheopen(sharp_data_url, output_path), parse_dates=[1])

if __name__ == "__main__":

    data = load_sharp_data();
    _assign_labels(data);

