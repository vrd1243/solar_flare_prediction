"""
This module provides an interface to load solar flare data for prediction.
Functions that pull data from online will cache files in the
data directory (~/.flare_prediction_data).

SHARP data in this module refers to a table like structure (csv, DataFrame)
with each row containing data recorded for some SHARP at some time.
Each row has a HARP number ("HARPNUM"), time recored ("T_REC"), and
a NOAA active region number ("NOAA_AR"). By default, the 16 spaceweather
quantities outlined at http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm
are also included. The final included column is a base URL ("base_url"),
which provides a way to construct a URL to download segment data.

If you prepend http://jsoc.stanford.edu to the base URL and then append
<segment>.fits to that, you'll get a valid URL to download a segment
ex: base URL = /SUM97/D1037535322/S00000/, segment = Br,
final URL = http://jsoc.stanford.edu/SUM97/D1037535322/S00000/Br.fits

Other notes about the SHARP data is that times recorded for a specific HARP
are at an hourly cadence, unless data was missing (in that case, data might
be farther than an hour apart.

The flare catalogue lists all M and X class flares associated with some HARP,
with columns for class, start time, end time, latitude, and NOAA AR number.

Invoking this module as script will create a local copy of the SHARP data.
It takes around ~2 hours (on my computer) to query all the data and assign labels.

author: Justin Cai
date: 2018-06-11
"""
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

__all__ = ["load_flare_catalogue", "load_sharp_data", "sharp_params", "get_segment_filepath", "get_segment"]

logging.basicConfig(level=logging.INFO)

# This lists all the HARP numbers and what NOAA AR(s) they are associated with.
# We can use this to find which HARP numbers to query instead of enumerating
# over the range of HARP numbers to speed things up.
harp_to_noaa_url = "http://jsoc.stanford.edu/doc/data/hmi/harpnum_to_noaa/all_harps_with_noaa_ars.txt"
flare_catalogue_url = "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/" \
                      "solar-flares/x-rays/goes/xrs/goes-xrs-report_%d.txt"
sharp_data_url = "https://www.dropbox.com/s/boyk2hpcw1bv1xt/data.csv?dl=1"
data_dir = os.path.expanduser("~/.flare_prediction_data")

output_path = os.path.join(data_dir, "data.csv")
failed_path = os.path.join(data_dir, "failed_%s.txt")

sharp_params = [
"USFLUX",
    "MEANGAM",
    "MEANGBT",
    "MEANGBZ",
    "MEANGBH",
    "MEANJZD",
    "TOTUSJZ",
    "MEANALP",
    "MEANJZH",
    "TOTUSJH",
    "ABSNJZH",
    "SAVNCPP",
    "MEANPOT",
    "TOTPOT",
    "MEANSHR",
    "SHRGT45",
    "AREA_ACR",
]


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


def _parallel_foreach(consumer_q, consumer_function, n_threads):
    """
    Apply consumer_function to each item of consumer_q using n_threads in parallel.
    :param consumer_q: a queue of items
    :param consumer_function: a function taking an item of the queue as a parameter
    :param n_threads: number of threads
    """

    def _consumer():
        while True:
            try:
                item = consumer_q.get_nowait()
            except Empty:
                return

            consumer_function(item)
            consumer_q.task_done()

    threads = [Thread(target=_consumer) for _ in range(n_threads)]
    for t in threads:
        t.daemon = True
        t.start()

    consumer_q.join()


def load_sharp_data():
    """Load the SHARP data, and return it in a data frame."""
    return pd.read_csv(_cacheopen(sharp_data_url, output_path), parse_dates=[1])


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


def _query_data(n_threads, n_tries, time_between_attempts, dataset):
    """
    Query JSOC and create unlabeled SHARP data, and write a local copy.
    :param n_threads: number of threads to use while querying
    :param n_tries: number of attempts to use per query
    :param time_between_attempts: time between failed queries
    :param dataset: JSOC dataset to query from
    :return: dataframe of SHARP data
    """

    keys = ["HARPNUM", "T_REC", "NOAA_AR", "LAT_FWT", "LON_FWT"]
    keys += sharp_params

    key_str = ", ".join(keys)

    harp_q = Queue()
    fail_q = Queue()
    df_q = Queue()
    with _cacheopen(harp_to_noaa_url) as data:
        lines = data.read().splitlines()
        for line in lines[1:]:
            harp_num, ar = line.split()
            harp_q.put(harp_num)

    c = Client()

    def _query(harp):
        """
        Repeatedly pulls query strings from harp_q, putting
        their results into df_q, and putting the query string into fail_q
        if the query failed. The function exits when harp_q is empty.
        """
        ds = "%s[%s][]" % (dataset, harp)
        failed = True

        # For some reason, I've observed querying JSOC sometimes fails with connection refused or HTTP error 500.
        # Because of this, each thread repeatedly attempts to query, and if the operation fails
        # n_tries times, then push it to fail_q. The failed results are reported at the end.
        for i in range(n_tries):
            try:
                keydf, segdf = c.query(ds, key=key_str, seg="bitmap")
                df = keydf.assign(
                    T_REC=pd.to_datetime(keydf["T_REC"], format="%Y.%m.%d_%H:%M:%S_TAI"),
                    # Extract base url by removing the last len("bitmap.fits") characters of the string
                    base_url=segdf["bitmap"].str[:-len("bitmap.fits")]
                )
                df_q.put(df[(df["T_REC"].dt.hour - df["T_REC"].shift(1).dt.hour) != 0])
                logging.info("Success %s", ds)
                failed = False
                break
            except Exception:
                if i != n_tries - 1:
                    logging.warning("Failed %s, %d more attempts", ds, n_tries - i - 1)
                    sleep(time_between_attempts)
        if failed:
            logging.warning("Failed %s, no more attempts", ds)
            fail_q.put(ds)

    _parallel_foreach(harp_q, _query, n_threads)

    data = pd.concat(df_q.queue, ignore_index=True)
    data.to_csv(output_path, index=False)
    with open(failed_path % "queries", "w") as f:
        f.write("\n".join(fail_q.queue))
    return data


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

    data.to_csv(output_path, index=False)


def _download_segments(segments, n_threads, n_tries, time_between_attempts, basedir):
    """
    For each row in the SHARP data, download each segment specified.
    :param segments: list of segments to downloads
    :param n_threads: number of threads to use while downloads
    :param n_tries: number of attempts per row
    :param time_between_attempts: time between failed attempts
    :param basedir: filepath to the base download directory
    """
    data = load_sharp_data()
    url_q = Queue()
    for s in segments:
        data[s] = basedir + data.index.astype(str) + (".%s.fits" % s)
    for _, row in data.iterrows():
        for s in segments:
            url_q.put((get_segment_filepath(s, basedir, row["T_REC"], row["HARPNUM"], "fits"),
                       "http://jsoc.stanford.edu%s%s.fits" % (row["base_url"], s)))

    fail_q = Queue()

    def _download(tup):
        """
        Download a segment.
        :param tup: a tuple of path and url
        """
        failed = True
        path, url = tup
        for i in range(5):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with urlopen(url) as infile, open(path, "wb") as outfile:
                    outfile.write(infile.read())
                logging.info("Success %s", url)
                failed = False
                break
            except Exception as e:
                if i != n_tries - 1:
                    logging.warning("%r", e)
                    logging.warning("Failed %s, %d more attempts", url, n_tries - i - 1)
                    sleep(time_between_attempts)
        if failed:
            logging.warning("Failed %s, no more attempts", url)
            fail_q.put(url)

    logging.info("starting")
    _parallel_foreach(url_q, _download, n_threads)
    with open(failed_path % "downloads", "w") as f:
        f.write("\n".join(fail_q.queue))
    return data


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", metavar="cmd", help="abc\na")
    subparsers.required = True

    fmt = argparse.ArgumentDefaultsHelpFormatter
    query = subparsers.add_parser("query", formatter_class=fmt,
                                  description="Query JSOC to get unlabeled SHARP data.")
    query.add_argument("--n_threads", type=int, default=10, help="Number of threads to use while querying")
    query.add_argument("--n_attempts", type=int, default=10, help="Number of attempt per query")
    query.add_argument("--attempt_timeout", type=float, default=5, help="Time between failed attempts")
    query.add_argument("--dataset", type=str, default="hmi.sharp_cea_720s", help="Dataset to query")
    query.set_defaults(func=lambda a: _query_data(a.n_threads, a.n_attempts, a.attempt_timeout, a.dataset))

    label = subparsers.add_parser("label", description="Assign labels to SHARP data.")
    label.set_defaults(func=lambda a: _assign_labels(load_sharp_data()))

    make = subparsers.add_parser("make", formatter_class=fmt,
                                 description="Make a copy of the full, labeled SHARP data.")
    make.add_argument("--n_threads", type=int, default=10, help="Number of threads to use while querying")
    make.add_argument("--n_attempts", type=int, default=10, help="Number of attempt per query")
    make.add_argument("--attempt_timeout", type=float, default=10, help="Time between failed attempts")
    make.add_argument("--dataset", type=str, default="hmi.sharp_cea_720s", help="Dataset to query")
    make.set_defaults(func=lambda a: _assign_labels(_query_data(a.n_threads, a.n_attempts,
                                                                a.attempt_timeout, a.dataset)))

    download = subparsers.add_parser("download", formatter_class=fmt,
                                     description="Download segments of SHARP data, additionally adding "
                                                 "their file path to the data.")
    download.add_argument("basedir")
    download.add_argument("segments", nargs="+")
    download.add_argument("--n_threads", type=int, default=10, help="Number of threads to use while downloading")
    download.add_argument("--n_attempts", type=int, default=10, help="Number of attempt per download")
    download.add_argument("--attempt_timeout", type=float, default=10, help="Time between failed downloads")
    download.set_defaults(func=lambda a: _download_segments(a.segments, a.n_threads, a.n_attempts,
                                                            a.attempt_timeout, a.basedir))

    args = parser.parse_args()
    print(args)
    args.func(args)
