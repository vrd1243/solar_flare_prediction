
import json, urllib, numpy as np, matplotlib.pylab as plt, matplotlib.ticker as mtick, requests
import sunpy.map
import drms
from astropy.io import fits
from sunpy.cm import color_tables as ct
#import sunpy.wcs as wcs
#import sunpy.coordinates as coord
import datetime
from datetime import datetime as dt_obj
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
#mpl.use('Agg')
import matplotlib.dates as mdates
import matplotlib.colors as mcol
from matplotlib.colors import LogNorm
import matplotlib.patches as ptc
from matplotlib.dates import *
import math
import h5py
import hickle as hkl
import drms
from sunpy.time import parse_time
from astropy.time import Time, TimeDelta, TimeDatetime
import pandas as pd

def get_hmi(keyword):
    c = drms.Client()
    keys, segments = c.query(keyword, key=drms.const.all, seg='magnetogram')
    url_hmi = 'http://jsoc.stanford.edu' + segments.magnetogram[0]   # add the jsoc.stanford.edu suffix to the segment name
    photosphere_full_image = fits.open(url_hmi)                   # download the image data
    photosphere_full_image.verify('fix')
    return keys, segments, photosphere_full_image

def extract_hmi_sdo(harpnum, hmi1_keyword, hmi2_keyword, harp_to_noaa_dict, flares, labels_writer, tds):
    
    kind = ['M', 'X']
    print(hmi1_keyword, hmi2_keyword)
    c = drms.Client()
    wavelengths=['131', '171', '193', '304', '1600']
    #wavelengths = []
    channels = ['aia.lev1_euv_12s', 'aia.lev1_euv_12s', 'aia.lev1_euv_12s', 'aia.lev1_euv_12s', 'aia.lev1_uv_24s']
    #time = '2012.03.06_23:29:06_TAI' 
    
    keys_hmi1, segments_hmi1, image_hmi1 = get_hmi('hmi.sharp_720s[{}][{}]'.format(harpnum, hmi1_keyword))
    url_hmi1 = 'http://jsoc.stanford.edu' + segments_hmi1.magnetogram[0]   # add the jsoc.stanford.edu suffix to segment name
    image_hmi1  = fits.open(url_hmi1)                           # download the data

    keys_hmi2, segments_hmi2, image_hmi2 = get_hmi('hmi.sharp_720s[{}][{}]'.format(harpnum, hmi2_keyword)) 
    url_hmi2 = 'http://jsoc.stanford.edu' + segments_hmi2.magnetogram[0]   # add the jsoc.stanford.edu suffix to segment name
    image_hmi2  = fits.open(url_hmi2)                           # download the data

    XDIM_HMI = image_hmi1[1].data.shape[1]
    YDIM_HMI = image_hmi1[1].data.shape[0]
    
    ts_hmi1 = Time(parse_time(keys_hmi1.T_REC[0])).tai.to_datetime();
    ts_hmi2 = Time(parse_time(keys_hmi2.T_REC[0])).tai.to_datetime();

    if (keys_hmi1.CROTA2[0] > 5.0):
            image_hmi1[1].data = np.rot90(image_hmi1[1].data,2)

    if (keys_hmi2.CROTA2[0] > 5.0):
            image_hmi2[1].data = np.rot90(image_hmi2[1].data,2)

    hmi_delta = (ts_hmi2 - ts_hmi1).total_seconds()
    
    #print(ts_hmi1 > ts_hmi1 + TimeDelta(48, format='sec'))
    
    time = ts_hmi1;
    
    while time < ts_hmi2:
        time_str = time.strftime("%Y.%m.%d_%H:%M:%S_TAI")
        print(time_str)
        
        # Extract the AIA wavelengths
        for wavelength in wavelengths:
            keys_aia, segments = c.query('aia.lev1[{}][?WAVELNTH={}?]'.format(time_str + '/48s', wavelength), key='T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2', seg='image_lev1')
            ts_aia = Time(keys_aia.T_REC[0]);
            #print(aia_keyword, ts_aia.tai)
            url_aia = 'http://jsoc.stanford.edu' + segments.image_lev1[0]   # add the jsoc.stanford.edu suffix to the segment name
            chromosphere_image = fits.open(url_aia) 

            ratio = (keys_hmi1.CDELT1[0])/(keys_aia.CDELT1[0])

            chromosphere_image.verify("fix")
            if (keys_aia.CROTA2[0] > 5.0):
                    chromosphere_image[1].data = np.rot90(chromosphere_image[1].data,2)
            
            aia_delta = (time - ts_hmi1).total_seconds()

            hmi_x_at_aia = keys_hmi1.CRPIX1[0] + (keys_hmi2.CRPIX1[0] - keys_hmi1.CRPIX1[0]) * (aia_delta / hmi_delta)
            hmi_y_at_aia = keys_hmi1.CRPIX2[0] + (keys_hmi2.CRPIX2[0] - keys_hmi1.CRPIX2[0]) * (aia_delta / hmi_delta)

            y1 = int(np.rint(2048. + hmi_y_at_aia*(ratio) - YDIM_HMI*(ratio)))
            y2 = int(np.rint(2048. + hmi_y_at_aia*(ratio)))
            x1 = int(np.rint(2048. + hmi_x_at_aia*(ratio) - XDIM_HMI*(ratio)))
            x2 = int(np.rint(2048. + hmi_x_at_aia*(ratio)))

            #sdoaiacmap = plt.get_cmap('sdoaia{}'.format(wavelength))
            subdata = chromosphere_image[1].data[y1:y2, x1:x2]
            
            chromosphere_image[1].header['START_X'] = x1;
            chromosphere_image[1].header['START_Y'] = y1;
            chromosphere_image[1].header['END_X'] = x2;
            chromosphere_image[1].header['END_Y'] = y2;
        
            t0 = time
            for td in tds:
                t1 = t0 + td
                df_td = flares[flares["start"].between(t0, t1)]
                #df_td = df_td[df_td["noaa"] == int(harpnum)]
                df_td = df_td[df_td["noaa"].isin(harp_to_noaa_dict[int(harpnum)])]

                hours = td.total_seconds() / 3600
                for k in kind:
                   chromosphere_image[1].header["%s_flare_in_%dh" % (k, hours)] = int(len(df_td[df_td["class"] == k]) != 0);
            
            header = dict();
            
            for h in chromosphere_image[1].header:
                header[h] = chromosphere_image[1].header[h];
            
            np.savez('/srv/data/varad/data/sdo_dataset/AIA_header_{}_{}_{}.npz'.format(harpnum, time_str, wavelength), header, allow_pickle=True)
            np.savez('/srv/data/varad/data/sdo_dataset/AIA_image_{}_{}_{}.npz'.format(harpnum, time_str, wavelength), subdata);
            
        # Extract the line-of-sight magnetogram. 
        keys_hmi_los, segments = c.query('hmi.M_45s[{}]'.format(time_str + '/45s'), key='T_REC,CROTA2,CDELT1,CDELT2,CRPIX1,CRPIX2,CRVAL1,CRVAL2', seg = 'magnetogram')
        url_hmi_los = 'http://jsoc.stanford.edu' + segments.magnetogram[0]   # add the jsoc.stanford.edu suffix to the segment name
        hmi_los_image = fits.open(url_hmi_los) 
        ratio = (keys_hmi1.CDELT1[0])/(keys_hmi_los.CDELT1[0])

        hmi_los_image.verify("fix")
        if (keys_hmi_los.CROTA2[0] > 5.0):
                hmi_los_image[1].data = np.rot90(hmi_los_image[1].data,2)
        
        hmi_los_delta = (time - ts_hmi1).total_seconds()

        hmi_x_at_hmi_los = keys_hmi1.CRPIX1[0] + (keys_hmi2.CRPIX1[0] - keys_hmi1.CRPIX1[0]) * (hmi_los_delta / hmi_delta)
        hmi_y_at_hmi_los = keys_hmi1.CRPIX2[0] + (keys_hmi2.CRPIX2[0] - keys_hmi1.CRPIX2[0]) * (hmi_los_delta / hmi_delta)

        y1 = int(np.rint(2048. + hmi_y_at_hmi_los*(ratio) - YDIM_HMI*(ratio)))
        y2 = int(np.rint(2048. + hmi_y_at_hmi_los*(ratio)))
        x1 = int(np.rint(2048. + hmi_x_at_hmi_los*(ratio) - XDIM_HMI*(ratio)))
        x2 = int(np.rint(2048. + hmi_x_at_hmi_los*(ratio)))

        #sdoaiacmap = plt.get_cmap('sdoaia{}'.format(wavelength))
        subdata = hmi_los_image[1].data[y1:y2, x1:x2]
        
        hmi_los_image[1].header['START_X'] = x1;
        hmi_los_image[1].header['START_Y'] = y1;
        hmi_los_image[1].header['END_X'] = x2;
        hmi_los_image[1].header['END_Y'] = y2;
        
        header = dict();
        
        for h in hmi_los_image[1].header:
            header[h] = hmi_los_image[1].header[h];
        
        np.savez('/srv/data/varad/data/sdo_dataset/HMI_LOS_header_{}_{}.npz'.format(harpnum, time_str), header, allow_pickle=True)
        np.savez('/srv/data/varad/data/sdo_dataset/HMI_LOS_image_{}_{}.npz'.format(harpnum, time_str), subdata);
        
        rowdict = {'timestamp': time_str}
        t0 = time
        for td in tds:
            t1 = t0 + td
            df_td = flares[flares["start"].between(t0, t1)]
            #df_td = df_td[df_td["noaa"] == int(harpnum)]
            df_td = df_td[df_td["noaa"].isin(harp_to_noaa_dict[int(harpnum)])]

            hours = td.total_seconds() / 3600
            for k in kind:
                rowdict["%s_flare_in_%dh" % (k, hours)] = int(len(df_td[df_td["class"] == k]) != 0)    
                hmi_los_image[1].header["%s_flare_in_%dh" % (k, hours)] = int(len(df_td[df_td["class"] == k]) != 0);
        
        labels_writer.writerow(list(rowdict.values()))

        time = time + datetime.timedelta(seconds=48);

def main():
    extract_hmi_sdo();

if __name__ == '__main__':
    main();
