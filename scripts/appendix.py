#!/usr/bin/env python
# This file plots all the correction maps for 1tff (Fig 9 to 12)

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

all_moments_ = moments()
fmp = first_moment_plotter()

def A1(): # makes the moment map summary for 1tff 
    path='mom_map_sum_1tff/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    fmp.without_colorbar(data=all_moments_.first_mom_isolated_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4]+'_0', angle=LOS_labels[0])
    fmp.without_colorbar(data=all_moments_.first_mom_isolated_ideal[1], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_0', angle=LOS_labels[0])
    fmp.without_colorbar(data=all_moments_.first_mom_isolated_ideal[2], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_0', angle=LOS_labels[0])

    fmp.without_colorbar(data=all_moments_.zero_mom_isolated_co10[0], text=img_names[0], file=outpath+path+files_ideal_npy[7][:-4]+'_1', angle=LOS_labels[1])
    fmp.without_colorbar(data=all_moments_.zero_mom_isolated_co10[1], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_1', angle=LOS_labels[1])
    fmp.without_colorbar(data=all_moments_.zero_mom_isolated_co10[2], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_1', angle=LOS_labels[1])

    fmp.without_colorbar(data=all_moments_.second_mom_isolated_co21[0], text=img_names[0], file=outpath+path+files_ideal_npy[4][:-4]+'_2', angle=LOS_labels[2])
    fmp.without_colorbar(data=all_moments_.second_mom_isolated_co21[1], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_2', angle=LOS_labels[2])
    fmp.without_colorbar(data=all_moments_.second_mom_isolated_co21[2], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_2', angle=LOS_labels[2])
