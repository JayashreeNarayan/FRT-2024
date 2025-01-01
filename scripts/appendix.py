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

from util_scripts.main import *
from util_scripts.all_functions import *
from util_scripts.universal_variables import *

all_moments_ = moments()
fmp = first_moment_plotter()

def A1(): # makes the moment map summary for 1tff 
    path='appendix/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[0][:-4]+'_00', 
                                  angle=LOS_labels[0], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[0], coords_of_fig='00', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_ideal[1], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4]+'_10', 
                                  angle=LOS_labels[1], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[3], coords_of_fig='10', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_ideal[2], text=img_names[0], file=outpath+path+files_ideal_npy[2][:-4]+'_20', 
                                  angle=LOS_labels[2], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[2], coords_of_fig='20', tot_panels=9)

    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[0][:-4]+'_01', 
                                  angle=LOS_labels[0], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[0], coords_of_fig='01', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co10[1], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_11', 
                                  angle=LOS_labels[1], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[3], coords_of_fig='11', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co10[2], text=img_names[1], file=outpath+path+files_co10_npy[2][:-4]+'_21', 
                                  angle=LOS_labels[2], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[2], coords_of_fig='21', tot_panels=9)

    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[0][:-4]+'_02', 
                                  angle=LOS_labels[0], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[0], coords_of_fig='02', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co21[1], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_12', 
                                  angle=LOS_labels[1], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[3], coords_of_fig='12', tot_panels=9)
    fmp.without_colorbar_appendix(data=all_moments_.first_mom_isolated_co21[2], text=img_names[2], file=outpath+path+files_co21_npy[2][:-4]+'_22', 
                                  angle=LOS_labels[2], xlabel_own = xyzlabels[1], ylabel_own = xyzlabels[2], coords_of_fig='22', tot_panels=9)

A1()