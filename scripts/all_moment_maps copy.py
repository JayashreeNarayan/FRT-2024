#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":    
    class moment_map():
        def __init__(self):
            self.all_sigmas_before=[]
            self.all_sigmas=[]
            self.ideal_1tff=[]


    class correction_factors:
        # For the correction factor maps, all after isolation
        ideal_1tff = []
        ideal_SE = []
        CO_10_1tff = []
        CO_10_SE = []
        CO_21_1tff = []
        CO_21_SE = []

        # vmin and vmax for correction factors
        vmin_correc = -1000
        vmax_correc = 1000
        ylim_min = 1.e-6
        ylim_max = 100

        # For the correction factor PDFs, all after isolation
        PDF_correction_obj = []
        correction_sigmas = []
        correction_labels = [r"CO (1-0) at $1~t_\mathrm{ff}$", r"CO (1-0) at $1.2~t_\mathrm{ff}$", r"CO (2-1) at $1~t_\mathrm{ff}$", r"CO (2-1) at $1.2~t_\mathrm{ff}$"]
        correction_xlabel = "Correction factors"

        correction_bins_values = [-1000, -900, -800, -700, -600, -500, -400, -300, -200, -100 ,0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    class pdf:
        # For the PDFs
        xmin=-0.45
        xmax=+0.45
        ymin=1.e-2
        ymax=5.e2
        xfit = np.linspace(xmin, xmax, 500)

        img_PDF_names_xpos = 0.02
        img_PDF_names_ypos = 0.85

        LOS_PDF_labels_xpos = 0.82
        LOS_PDF_labels_ypos = 0.75

        PDF_obj_isolated = []
        PDF_obj_SE = []

        # PDF objects and sigmas
        PDF_obj = []

    class simend:
        vmax_0_SE = 6
        vmax_2_SE = 0.6

    class ft:        
        # Fourier labels
        FT_xy_labels = [r"$k$", r"$P_\mathrm{tot}$"]
        angles = [0, 45, 90]
        FTdata = []
        FTdata_raw = []
        FTdata_SE = []

        # defining kmin and kmax for FT spectra graphs:
        kmin = 3
        kmax = 40

    class skew_kurtosis:
        skewness_isolated = []
        kurtosis_isolated = []
        skewness_SE_before = []
        kurtosis_SE_before = []
        skewness_SE_after = []
        kurtosis_SE_after = []
        skewness = []
        kurtosis = []

    class sigma:
        all_sigmas = []
        all_sigmas_before = []
        sigma = []
        sigma_SE = []
        sigma_isolated = []
