#!/usr/bin/env python

import numpy as np
import cfpack as cfp
from cfpack import print, stop, hdfio
import argparse
import os

def zeromoment(PPV):
    PPV = np.load(PPV)
    K = np.sum(PPV,axis=2)*0.05
    return K

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot files')

    choices = ['cfp', 'flashplotlib', 'PPV']
    parser.add_argument('-a', '--action', metavar='action', nargs='*', default=choices, choices=choices,
                        help='choice: between flashplotlib plotting (flash) and cfpack plotting (cfp)')

    args = parser.parse_args()

    # set paths and create plot directory
    path = "../Data/"
    outpath = "../plots/"
    if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

    # set some global option/variables
    vmin = -0.4
    vmax = +0.4

    # loop through chosen actions
    for action in args.action:

        # cfpack plotting
        if action == choices[0]:
            files = ["FMM_0.0_0.0.npy", "FMM_90.0_0.0.npy"]
            for file in files:
                data = np.load(path+"Data_1tff/Othin/"+file)
                data = np.flipud(data).T
                cfp.plot_map(data, cmap='seismic', cmap_label=r"$v$ (km/s)", vmin=vmin, vmax=vmax, save=outpath+file[:-3]+"pdf")

        # flashplotlib
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -d vel"+dir+" -nolog -cmap seismic -mw -direction "+dir+" -outtype pdf -outdir "+outpath+" -vmin "+str(vmin*1e5)+" -vmax "+str(vmax*1e5)
                cfp.run_shell_command(cmd)

        #zero moment maps from PPV cubes
        if action == choices[2]:
            files = ["PPV_0_0.npy","PPV_90_0.npy"]
            for file in files:
                data = zeromoment(path+"Data_1tff/"+file)
                cfp.plot_map(data, cmap='seismic', cmap_label=r"$Density (g/cm^3)$", vmin=vmin, vmax=vmax, save=outpath+file[:-3]+"pdf")
                
