#!/bin/python

import numpy as np
from PIL import Image
import sys
from homology import Intervals
from matplotlib import pyplot as plt
from astropy.io import fits


filepath = sys.argv[1]
#image = Image.open(filepath)
#data = np.array(image)
hdu = fits.open(sys.argv[1])
hdu[1].verify('fix')
data = hdu[1].data
data[np.where(data < 0)] = 0

epsilons = np.linspace(np.min(data), np.max(data), 20)
intervals = Intervals(data, '1', epsilons)

plt.figure(figsize=(10,10), dpi=300)
plt.xlim((0,3000))
plt.ylim((0,3000))
plt.xlabel('Birth Threshold Flux', fontsize=20)
plt.ylabel('Death Threshold Flux', fontsize=20)
y = x = np.linspace(0,3000,100)
plt.plot(x,y, color='black')
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.scatter(intervals.birth_time, intervals.death_time,s=10)
plt.savefig('cubical-PD.png')
