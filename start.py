import matplotlib.pyplot as plt
import numpy as np
import cfpack as cfp
import argparse
import os

os.chdir("Data/Data_SimEnd/Othin")

parser = argparse.ArgumentParser(description='Plot files')

parser.add_argument('npy_file_names', metavar='N', type=str, nargs='+',
                    help='file name for cfpack to plot them')

args = parser.parse_args()

data=np.load(args.npy_file_names[0])
cfp.plot_map(data, show=True, cmap = 'seismic')


