#!/usr/bin/env python
# This file plots moment map summary at 1tff (Fig. 1) and at SE (Fig. 5)

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
    path='mom_map_sum_1tff/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    fmp.with_colorbar(data=moments_45_.first_mom_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[1][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_fmp')
    fmp.with_colorbar(data=moments_45_.first_mom_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_fmp')

    zmp.with_colorbar(data=moments_45_.zero_mom_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[7][:-4])
    zmp.with_colorbar(data=moments_45_.zero_mom_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_zmp')
    zmp.with_colorbar(data=moments_45_.zero_mom_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_zmp')

    smp.with_colorbar(data=moments_45_.second_mom_ideal[0], text=img_names[0], file=outpath+path+files_ideal_npy[4][:-4])
    smp.with_colorbar(data=moments_45_.second_mom_co10[0], text=img_names[1], file=outpath+path+files_co10_npy[1][:-4]+'_smp')
    smp.with_colorbar(data=moments_45_.second_mom_co21[0], text=img_names[2], file=outpath+path+files_co21_npy[1][:-4]+'_smp')

def mom_map_sum_SE(): # makes the moment map summary for 1.2tff 
    path='mom_map_sum_SE/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    fmp.with_colorbar(data=moments_45_.first_mom_ideal[1], text=img_names[0], file=outpath+path+files_ideal_SE_npy[0][:-4])
    fmp.with_colorbar(data=moments_45_.first_mom_co10[1], text=img_names[1], file=outpath+path+file_co10_SE_npy[0][:-4]+'_fmp')
    fmp.with_colorbar(data=moments_45_.first_mom_co21[1], text=img_names[2], file=outpath+path+file_co21_SE_npy[0][:-4]+'_fmp')

    zmp.with_colorbar(data=moments_45_.zero_mom_ideal[1], text=img_names[0], file=outpath+path+files_ideal_SE_npy[2][:-4])
    zmp.with_colorbar(data=moments_45_.zero_mom_co10[1], text=img_names[1], file=outpath+path+file_co10_SE_npy[0][:-4]+'_zmp')
    zmp.with_colorbar(data=moments_45_.zero_mom_co21[1], text=img_names[2], file=outpath+path+file_co21_SE_npy[0][:-4]+'_zmp')

    smp.with_colorbar(data=moments_45_.second_mom_ideal[1], text=img_names[0], file=outpath+path+files_ideal_SE_npy[1][:-4])
    smp.with_colorbar(data=moments_45_.second_mom_co10[1], text=img_names[1], file=outpath+path+file_co10_SE_npy[0][:-4]+'_smp')
    smp.with_colorbar(data=moments_45_.second_mom_co21[1], text=img_names[2], file=outpath+path+file_co21_SE_npy[0][:-4]+'_smp')
    

mom_map_sum_1tff()
mom_map_sum_SE()
