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
zmp = zeroth_moment_plotter()
smp = second_moment_plotter()

def mom_map_sum_1tff(): # makes the moment map summary for 1tff 
    fmp.with_colorbar(data=moments_45_.first_mom_ideal[0], text=img_names[0], file=outpath+files_ideal_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_co10[0], text=img_names[1], file=outpath+file_co10_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_co21[0], text=img_names[2], file=outpath+file_co21_npy[1][:-4])

    zmp.with_colorbar(data=moments_45_.zero_mom_ideal[0], text=img_names[0], file=outpath+files_ideal_npy[1][:-4])
    zmp.with_colorbar(data=moments_45_.zero_mom_co10[0], text=img_names[1], file=outpath+file_co10_npy[1][:-4])
    zmp.with_colorbar(data=moments_45_.zero_mom_co21[0], text=img_names[2], file=outpath+file_co21_npy[1][:-4])

    fmp.with_colorbar(data=moments_45_.first_mom_ideal, text=img_names[0], file=outpath+files_ideal_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_ideal, text=img_names[0], file=outpath+files_ideal_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_ideal, text=img_names[0], file=outpath+files_ideal_npy[1][:-4])

    # doing the same as above for 1.2tff
    

mom_map_sum_1tff()