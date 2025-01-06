#!/usr/bin/env python
# This file plots moment map summary at 1tff (Fig. 2) and at SE (Fig. 6)

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

moments_45_ = moments_45()
fmp = first_moment_plotter()

def turb_isolation_1tff(): # makes the turbulence isolation maps for 1tff - 9 images
    path='turb_isolation_1tff/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    # before row
    fmp.without_colorbar(data=moments_45_.first_mom_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4]+'_before', coords_of_fig='00')
    fmp.without_colorbar(data=moments_45_.first_mom_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_before', coords_of_fig='01')
    fmp.without_colorbar(data=moments_45_.first_mom_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_before', coords_of_fig='02')

    # smoothed row
    fmp.without_colorbar(data=moments_45_.first_mom_smooth_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4]+'_smooth', coords_of_fig='10')
    fmp.without_colorbar(data=moments_45_.first_mom_smooth_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_smooth', coords_of_fig='11')
    fmp.without_colorbar(data=moments_45_.first_mom_smooth_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_smooth', coords_of_fig='12')

    # after row
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4]+'_isol', coords_of_fig='20')
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_isol', coords_of_fig='21')
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_isol', coords_of_fig='22')
    
    # common colorbar 
    fmp.colorbar(panels=3)

def turb_isolation_SE(): # makes the turbulence isolation maps for SE - 3 images
    path='turb_isolation_SE/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_ideal[1], text=img_names[0], file=outpath+path+files_ideal_SE_npy[0][:-4]+'_isol', coords_of_fig='20')
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_co10[1], text=img_names[1], file=outpath+path+file_co10_SE_npy[0][:-4]+'_isol', coords_of_fig='21')
    fmp.without_colorbar(data=moments_45_.first_mom_isolated_co21[1], text=img_names[2], file=outpath+path+file_co21_SE_npy[0][:-4]+'_isol', coords_of_fig='22')

turb_isolation_1tff()
turb_isolation_SE()
