#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import cfpack as cfp
from cfpack import print, stop
import argparse
import os
import subprocess

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot files')

    choices = ['cfp', 'flash1x', 'flash1y', 'flash1z']
    parser.add_argument('-a', '--action', metavar='action', nargs='*', default=choices, choices=choices,
                        help='choice: between flashplotlib plotting (flash) and cfpack plotting (cfp); flash options: flash1x (FMMx); flash1y (FMMy) and flash1z (FMMz)')

    parser.add_argument('-f', '-file_names', metavar='file', type=str, nargs='+', help='file name for cfpack to plot them;  file names: FMM_0.0_0.0.npy, FMM_90.0_0.0.npy, ChemoMHD_hdf5_plt_cnt_0000')

    args = parser.parse_args()

    for action in args.action:

        if action == choices[0]:
            stop()
            os.chdir("Data/Data_SimEnd/Othin")
            data=np.load(args.file_names[0])
            cfp.plot_map(data, show=True, cmap = 'seismic')

        if action == choices[1]:
            os.chdir("Data/Flashfile")
            data=args.file_names[0]
            subprocess.run(["flashplotlib.py","-i",data, "-d", "velx", "-nolog", "-cmap","seismic","-mw","-direction", "x"])

        if action == choices[2]:
            os.chdir("Data/Flashfile")
            data=args.file_names[0]
            subprocess.run(["flashplotlib.py","-i",data, "-d", "vely", "-nolog", "-cmap","seismic","-mw","-direction", "y"])

        if action == choices[3]:
            os.chdir("Data/Flashfile")
            data=args.file_names[0]
            subprocess.run(["flashplotlib.py","-i",data, "-d", "velz", "-nolog", "-cmap","seismic","-mw","-direction", "z"])
