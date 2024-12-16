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
    
    def first_moment_calc(self):
        if files_ideal.key[:2]=='ZMM': self.zero_mom_ideal.append(files_ideal.value)


