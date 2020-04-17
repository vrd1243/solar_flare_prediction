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
from subprocess import call

def run_perseus(perseus_arr, betti = '0'):
    
    pid = os.getpid()
    infile = "perseus_in_{}.txt".format(pid)
    outfile = "perseus_out_{}.txt".format(pid)

    np.savetxt(infile , perseus_arr, fmt='%d');

    call(["./perseus", "cubtop", infile, outfile], stdout=open(os.devnull, 'wb'))

    intervals = np.loadtxt(outfile + "_" + betti + ".txt")
    intervals[intervals == -1] = np.nan

    for f in glob.glob(outfile + "_*.txt"):
          os.remove(f)
    
    os.remove(infile);
    
    return intervals;

class Intervals:
        """birth and death times for holes in the complex filtration"""
        def __init__(self, data, betti, epsilons):

                #print("Generating Intervals ...", data.shape)
                perseus_arr = np.array([2, data.shape[0], data.shape[1]]).reshape((-1,1));
                perseus_arr = np.concatenate((perseus_arr, data.T.reshape((-1,1))), axis=0);
                intervals = run_perseus(perseus_arr, betti);
                self.epsilons = epsilons;
                try:
                        self.birth_time = intervals[:, 0]
                        self.death_time = intervals[:, 1]
                except IndexError:
                        self.birth_time = []
                        self.death_time = []
                
        def interval_count(self):
                
                count = np.zeros(self.epsilons.shape[0])

                for idx, eps in enumerate(self.epsilons):
                    for birth, death in zip(self.birth_time, self.death_time):

                        if birth <= eps and death >= eps:
                            count[idx] += 1
        
                return count

class PD:
        """persistence diagram"""
        def __init__(self, intervals):
                #print("Generating PD ...")
                self.epsilons = intervals.epsilons
                self.lim = self.epsilons[-1]
                self.mortal, self.immortal = self._build(intervals)


        def _t_to_eps(self, t):
                return t;
                #return self.epsilons[int(t - 1)]

        @staticmethod
        def _get_multiplicity(birth_e, death_e=None):
                if death_e is None:
                        death_e = np.full_like(birth_e, -1)
                count = np.zeros_like(birth_e)
                for i, pt in enumerate(zip(birth_e, death_e)):
                        for scanner_pt in zip(birth_e, death_e):
                                if pt == scanner_pt:
                                        count[i] += 1
                return count

        def _build(self, intervals):
                birth_e_mor = []
                death_e_mor = []
                birth_e_imm = []
                
                for birth, death in zip(intervals.birth_time, intervals.death_time):
                        if np.isnan(death):                                                        # immortal
                                birth_e_imm.append(self._t_to_eps(birth))
                        else:                                                                                    # mortal
                                birth_e_mor.append(self._t_to_eps(birth))
                                death_e_mor.append(self._t_to_eps(death))
                
                count_mor = self._get_multiplicity(birth_e_mor, death_e_mor)
                mortal = np.asarray([birth_e_mor, death_e_mor, count_mor]).T
                
                count_imm = self._get_multiplicity(birth_e_imm)
                immortal = np.asarray([birth_e_imm, count_imm]).T
                
                if len(mortal):
                        # toss duplicates #
                        mortal = np.vstack({tuple(row) for row in mortal})
                
                if len(immortal):

                        # toss duplicates #
                        immortal = np.vstack({tuple(row) for row in immortal})
                
                return mortal, immortal


class PRF:
        """persistence rank function"""
        def __init__(self, pd):
                #print("Generating PRFs ...")
                self.epsilons = pd.epsilons
                self.data = self._build(pd)

        def _build(self, pd):

                num_div = len(self.epsilons)
                max_lim = pd.lim
                min_lim = 0

                x = y = np.linspace(min_lim, max_lim, num_div)
                xx, yy = np.meshgrid(x, y)

                grid_pts = list(zip(np.nditer(xx), np.nditer(yy)))
                zz = np.zeros(len(grid_pts))
                for i, grid_pt in enumerate(grid_pts):
                        if grid_pt[0] <= grid_pt[1]:
                                for pt in pd.mortal:
                                        if pt[0] <= grid_pt[0] and pt[1] >= grid_pt[1]:
                                                zz[i] += pt[2]
                                for pt in pd.immortal:
                                        if pt[0] <= grid_pt[0]:
                                                zz[i] += pt[1]
                        else:
                                zz[i] = -1
                zz = np.reshape(zz, xx.shape)
                return zz;
