#!/usr/bin/env python
# This file plots the power spectra of the moment maps at 1tff (Fig. 4) and at SE (Fig. 8) -- TOTAL 3 IMAGES

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

def FT_1tff_before(): # plots Fig. 6 top panel
    path='all_FTs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    FT_obj = [] 
    FT_obj.append(moments_45_.first_mom_ideal_FT_obj[0])
    FT_obj.append(moments_45_.first_mom_co10_FT_obj[0])
    FT_obj.append(moments_45_.first_mom_co21_FT_obj[0])
    fmp.FT_panel(FT_obj, file=outpath+path+"before_isolation.pdf", text=img_types[0])

def FT_1tff_after(): # plots Fig. 5 bottom panel
    path='all_FTs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    FT_obj = [] 
    FT_obj.append(moments_45_.first_mom_isolated_ideal_FT_obj[0])
    FT_obj.append(moments_45_.first_mom_isolated_co10_FT_obj[0])
    FT_obj.append(moments_45_.first_mom_isolated_co21_FT_obj[0])
    fmp.FT_panel(FT_obj, file=outpath+path+"after_isolation.pdf", text=img_types[1])

def FT_SE_after(): # plots Fig. 7
    path='all_FTs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    FT_obj = [] 
    FT_obj.append(moments_45_.first_mom_isolated_ideal_FT_obj[1])
    FT_obj.append(moments_45_.first_mom_isolated_co10_FT_obj[1])
    FT_obj.append(moments_45_.first_mom_isolated_co21_FT_obj[1])
    fmp.FT_panel(FT_obj, file=outpath+path+"after_isolation_SE.pdf", text=img_types[1])
    
FT_1tff_before()
FT_1tff_after()
FT_SE_after()
