#!/usr/bin/env python
# This file plots moment map summary at 1tff (Fig. 1)

import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
from scipy import stats as st
from astropy import constants as c

from main import *
from all_functions import *
from universal_variables import *

moments_45_ = moments_45()
fmp = first_moment_plotter()

def mom_map_sum_1tff(): # makes the moment map summary for 1tff and 1.2tff
    print(files_ideal_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_ideal, text=img_names[0], file=outpath+files_ideal_npy[1][:-4])
    

mom_map_sum_1tff()