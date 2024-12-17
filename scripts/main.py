#!/usr/bin/env python
# This file is the classes of all the data

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *
from universal_variables import *

files_ideal={}
for file in files_ideal_npy:
    files_ideal.key = file
    files_ideal.value = np.load(path+"/Data_1tff/Othin/"+file)

class moments:
    def __init__(self):
        self.zero_mom_ideal=[]
        self.zero_mom_co10=[]
        self.zero_mom_co21=[]

        self.first_mom_ideal=[]
        self.first_mom_co10=[]
        self.first_mom_co21=[]

        self.second_mom_ideal=[]
        self.second_mom_co10=[]
        self.second_mom_co21=[]

        # for the ideal case
        for key, value in files_ideal.items():
            if key[:3]=='ZMM': self.zero_mom_ideal.append(rescale_data(value))
            elif key[:3]=='FMM': self.first_mom_ideal.append(value)
            elif key[:3]=='SMM': self.second_mom_ideal.append(value)
        
        # for the CO (1-0) case:
        for key, value in files_co10.items():
            if get_LOS(key) == 1:             
                self.zero_mom_co10.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co10.append(resize_45(first_moment(value, Vrange), '2D'))
                self.second_mom_co10.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co10.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co10.append(first_moment(value, Vrange))
                self.second_mom_co10.append(second_moment(value, Vrange))
        
        # for the CO (2-1) case:
        for key, value in files_co21.items():
            if get_LOS(key) == 1:             
                self.zero_mom_co21.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co21.append(resize_45(first_moment(value, Vrange), '2D'))
                self.second_mom_co21.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co21.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co21.append(first_moment(value, Vrange))
                self.second_mom_co21.append(second_moment(value, Vrange))
        
        
        
        



        


