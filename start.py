import matplotlib.pyplot as plt
import numpy as np
import cfpack as cfp
import argparse

parser = argparse.ArgumentParser(description='Plot files')

parser.add_argument('npy_file_names', metavar='N', type=str, nargs='+',
                    help='file name for cfpack to plot them')

#parser.add_argument('--sum', dest='accumulate', action='store_const',const=sum, default=max, help='sum the integers (default: find the max)')

args = parser.parse_args()
#print(args.accumulate(args.integers))

#print(args.file_names)
data=np.load(args.npy_file_names[0])
cfp.plot_map(data, show=True, cmap = 'seismic')


