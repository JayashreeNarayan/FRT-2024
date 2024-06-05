import matplotlib.pyplot as plt
import numpy as np
import cfpack as cfp
import argparse
import os
import subprocess

parser = argparse.ArgumentParser(description='Plot files')

parser.add_argument('action', metavar='action', nargs=1,
                    help='choice: between flashplotlib plotting (flash) and cfpack plotting (cfp); flash options: flash1x (FMMx); flash1y (FMMy) and flash1z (FMMz)')

parser.add_argument('file_names', metavar='file', type=str, nargs='+', help='file name for cfpack to plot them;  file names: FMM_0.0_0.0.npy, FMM_90.0_0.0.npy, ChemoMHD_hdf5_plt_cnt_0000')

args = parser.parse_args()
print(args)

if args.action[0] == 'cfp':
    os.chdir("Data/Data_SimEnd/Othin")
    data=np.load(args.file_names[0])
    cfp.plot_map(data, show=True, cmap = 'seismic')

if args.action[0] == 'flash1x':
    os.chdir("Data/Flashfile")
    data=args.file_names[0]
    subprocess.run(["flashplotlib.py","-i",data, "-d", "velx", "-nolog", "-cmap","seismic","-mw","-direction", "x"])

if args.action[0] == 'flash1y':
    os.chdir("Data/Flashfile")
    data=args.file_names[0]
    subprocess.run(["flashplotlib.py","-i",data, "-d", "vely", "-nolog", "-cmap","seismic","-mw","-direction", "y"])

if args.action[0] == 'flash1z':
    os.chdir("Data/Flashfile")
    data=args.file_names[0]
    subprocess.run(["flashplotlib.py","-i",data, "-d", "velz", "-nolog", "-cmap","seismic","-mw","-direction", "z"])

else:
    print("invalid choice")