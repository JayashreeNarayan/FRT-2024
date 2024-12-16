#!/usr/bin/env python
# This file is used to make all the moments - should take no time to run -- contains no plotting, only calculating

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *

def ideal_case():
    files = ["FMM_00.0_0.0.npy", "FMM_45.0_0.0.npy", "FMM_90.0_0.0.npy", "SMM_00.0_0.0.npy", "SMM_45.0_0.0.npy", "SMM_90.0_0.0.npy", "ZMM_00.0_0.0.npy", "ZMM_45.0_0.0.npy", "ZMM_90.0_0.0.npy"]
    for file in files:
        data = np.load(path+"/Data_1tff/Othin/"+file)
        
        if get_LOS(file) == 0: # theta is 0 degrees - along Z axis
            xlabel = xyzlabels[1] # y-axis on the horizontal
            ylabel = xyzlabels[0] # x-axis on the vertical
        
        if get_LOS(file) == 2: # theta is 90 degrees - along X axis 
            xlabel = xyzlabels[1] # y-axis on the horizontal
            ylabel = xyzlabels[2] # z-axis on the vertical 

        # for the files with 45 degrees -  we have to resize the data
        if get_LOS(file) == 1: # this means that theta is 45 degrees
            data = resize_45(data, "2D")
            xlabel = xyzlabels[1] # y-axis on the bottom 
            ylabel = xyzlabels[3] # combination of x and z on the vertical

            if file[:1] == "S": # Since the 2nd moment map needs different plot variables
                ideal_1tff_mom2 = data # for the correction factors