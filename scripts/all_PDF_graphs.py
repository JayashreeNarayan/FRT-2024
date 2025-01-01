#!/usr/bin/env python
# This file plots the PDFs of the moment maps at 1tff (Fig. 3) and at SE (Fig. 7) and for the correction maps (Fig. 13) -- TOTAL 4 IMAGES

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

def PDF_1tff_before(): # plots Fig. 5 top panel
    path='all_PDF_graphs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    PDF_obj = [] 
    PDF_obj.append(moments_45_.first_mom_ideal_PDF_obj[0])
    PDF_obj.append(moments_45_.first_mom_co10_PDF_obj[0])
    PDF_obj.append(moments_45_.first_mom_co21_PDF_obj[0])
    sigmas=[]
    sigmas.append(moments_45_.first_mom_ideal_sigma[0])
    sigmas.append(moments_45_.first_mom_co10_sigma[0])
    sigmas.append(moments_45_.first_mom_co21_sigma[0])
    fmp.PDF_panel(PDF_obj, sigmas, file=outpath+path+"before_isolation.pdf",text=img_types[0])

def PDF_1tff_after(): # plots Fig. 5 bottom panel
    path='all_PDF_graphs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    PDF_obj = [] 
    PDF_obj.append(moments_45_.first_mom_isolated_ideal_PDF_obj[0])
    PDF_obj.append(moments_45_.first_mom_isolated_co10_PDF_obj[0])
    PDF_obj.append(moments_45_.first_mom_isolated_co21_PDF_obj[0])
    sigmas=[]
    sigmas.append(moments_45_.first_mom_isolated_ideal_sigma[0])
    sigmas.append(moments_45_.first_mom_isolated_co10_sigma[0])
    sigmas.append(moments_45_.first_mom_isolated_co21_sigma[0])
    fmp.PDF_panel(PDF_obj, sigmas, file=outpath+path+"after_isolation.pdf", text=img_types[1])

def PDF_SE_after(): # plots Fig. 7
    path='all_PDF_graphs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    PDF_obj = [] 
    PDF_obj.append(moments_45_.first_mom_isolated_ideal_PDF_obj[1])
    PDF_obj.append(moments_45_.first_mom_isolated_co10_PDF_obj[1])
    PDF_obj.append(moments_45_.first_mom_isolated_co21_PDF_obj[1])
    sigmas=[]
    sigmas.append(moments_45_.first_mom_isolated_ideal_sigma[1])
    sigmas.append(moments_45_.first_mom_isolated_co10_sigma[1])
    sigmas.append(moments_45_.first_mom_isolated_co21_sigma[1])
    fmp.PDF_panel(PDF_obj, sigmas, file=outpath+path+"after_isolation_SE.pdf", text=img_types[1])

def PDF_corrections_before(): # plots Fig. 13
    path='all_PDF_graphs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    PDF_obj = [] 
    PDF_obj.append(moments_45_.first_mom_co10_correction_obj[0][1])
    PDF_obj.append(moments_45_.first_mom_co21_correction_obj[1][1])
    sigmas=[]
    sigmas.append(moments_45_.first_mom_co10_correction_obj[0][2])
    sigmas.append(moments_45_.first_mom_co21_correction_obj[1][2])
    fmp.PDF_corrections(PDF_obj, sigmas, file=outpath+path+"corrections_PDF_before.pdf", text=img_types[0])

def PDF_corrections_after(): # plots Fig. 13
    path='all_PDF_graphs/' # save these figures to a different subfolder
    if not os.path.isdir(outpath+path):
        cfp.run_shell_command('mkdir '+outpath+path)
    PDF_obj = [] 
    PDF_obj.append(moments_45_.first_mom_isolated_co10_correction_obj[0][1])
    PDF_obj.append(moments_45_.first_mom_isolated_co21_correction_obj[1][1])
    sigmas=[]
    sigmas.append(moments_45_.first_mom_isolated_co10_correction_obj[0][2])
    sigmas.append(moments_45_.first_mom_isolated_co21_correction_obj[1][2])
    fmp.PDF_corrections(PDF_obj, sigmas, file=outpath+path+"corrections_PDF_after.pdf", text=img_types[1])
    
#PDF_1tff_before()
#PDF_1tff_after()
#PDF_SE_after()
PDF_corrections_after()
PDF_corrections_before()
