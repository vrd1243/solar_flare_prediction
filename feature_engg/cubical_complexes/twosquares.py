#!/bin/python
import numpy as np

data = 100*np.ones((100,100));
data[25:75, 25:75] = 200;
data[40:60, 40:60] = 400;

perseus_arr = np.array([2, 100, 100]).reshape((-1,1));
perseus_arr = np.concatenate((perseus_arr, data.T.reshape((-1,1)) - np.min(data)+1), axis=0);

np.savetxt('twosquares_in.txt', perseus_arr, fmt='%d');
