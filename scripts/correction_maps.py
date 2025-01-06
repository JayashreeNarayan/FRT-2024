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

moments_45_ = moments_45()
cmp = correction_moment_plotter()

def correction_10():
    path='correction_maps/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    cmp.without_colorbar(data=moments_45_.first_mom_co10_correction_obj[0][0], text=correction_labels[0], file=outpath+path+"co_10_fmp", moment=1, coords_of_fig='00')
    cmp.without_colorbar(data=moments_45_.first_mom_co21_correction_obj[0][0], text=correction_labels[1], file=outpath+path+"co_21_fmp", moment=1, coords_of_fig='01')

    # for the before-after-isolated cases
    cmp.without_colorbar(data=moments_45_.first_mom_isolated_co10_correction_obj[0][0], text=correction_labels[0], file=outpath+path+"co_10_fmp_iso", moment=1, coords_of_fig='20')
    cmp.without_colorbar(data=moments_45_.first_mom_isolated_co21_correction_obj[0][0], text=correction_labels[1], file=outpath+path+"co_21_fmp_iso", moment=1, coords_of_fig='21')

    cmp.without_colorbar(data=moments_45_.zero_mom_co10_correction_obj[0][0], text=correction_labels[0], file=outpath+path+"co_10_zmp", moment=0,  coords_of_fig='10')
    cmp.without_colorbar(data=moments_45_.zero_mom_co21_correction_obj[0][0], text=correction_labels[1], file=outpath+path+"co_21_zmp", moment=0,  coords_of_fig='11')

    cmp.without_colorbar(data=moments_45_.second_mom_co10_correction_obj[0][0], text=correction_labels[0], file=outpath+path+"co_10_smp", moment=2, coords_of_fig='20')
    cmp.without_colorbar(data=moments_45_.second_mom_co21_correction_obj[0][0], text=correction_labels[1], file=outpath+path+"co_21_smp", moment=2, coords_of_fig='21')

    # common colorbars
    cmp.colorbar(panels=1, moment=0)
    cmp.colorbar(panels=1, moment=1)
    cmp.colorbar(panels=1, moment=2)

    # for the before-after iso one
    cmp.colorbar(panels=2, moment=1)

correction_10()