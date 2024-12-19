#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from universal_variables import *

def find_plots():
    if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

# Functions for FT
def secax_forward(x):
    L = 2
    return L/x

def secax_backward(x):
    L=2
    return L/x

# function that gets theta from the file name and decides which direction is the LOS
def get_LOS(file):
    theta = file[4:8]
    if theta == "00.0": return 0 # the no.s are based on LOS_labels
    if theta == "45.0": return 1
    if theta == "90.0": return 2

def fourier_spectrum(data):
    FT = cfp.get_spectrum(data)
    K = FT.get('k')
    P = FT.get('P_tot')
    return [K,P]

# function that divides data by its mean to avoid units
def rescale_data(data):
    return data/np.mean(data)

# function that creates the 0-moment map from a PPV cube with the velocity axis=2 and velocity channels in 'Vrange'
def zero_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = np.sum(PPV, axis=2) * dv
    return mom0

# Same as zero_moment, but for the 1st moment
def first_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = zero_moment(PPV, Vrange) # moment-0 for normalisation
    mom1 = np.sum(PPV*Vrange, axis=2) * dv
    return (mom1 / mom0)

# Same as first_moment, but for the 2nd moment
def second_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = (zero_moment(PPV, Vrange)) # moment-0 for normalisation
    mom2 = np.sum(PPV*Vrange**2, axis=2) * dv / mom0
    mom1 = np.sum(PPV*Vrange, axis=2) * dv / mom0
    mom2 = np.sqrt(mom2 - mom1**2)
    return mom2

# function that returns a Gaussian-smoothed version of the data - data being a 2D array
def smoothing(data):
    npix = data.shape[1]
    return cfp.gauss_smooth(data, fwhm=npix/2, mode='nearest').tolist()

def get_vmin_vmax_centred(data):
    minmax = np.max([-data.min(), data.max()])
    return -minmax, +minmax

def resize_45(data, choice):
    if choice == '2D':
        rescaled_data = np.zeros((128,128))
        for i in range(0,128):
            K = data[:,i]
            rescaled_data[i] = K[25:153]
        return np.array(rescaled_data)
    elif choice == '3D':
        A = list(range(0,25))
        B = list(range(128,153))
        rescaled_data = np.delete(data, A, 0)
        rescaled_data = np.delete(rescaled_data, B, 0)
        return rescaled_data
    else:
        A = list(range(0,26))
        B = list(range(127,153))
        rescaled_data = np.delete(data, A, 0)
        rescaled_data = np.delete(rescaled_data, B, 0)
        return rescaled_data

def PDF_img_names(i, sigma):
    sigma=cfp.round(sigma, 2, str_ret=True)
    img_names = ["ideal", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
    return img_names[i]+r": $\sigma$ = "+sigma+r"$~\mathrm{km\,s^{-1}}$"

def gauss_func(x, mean, sigma):
    gauss = 1.0/np.sqrt(2.0*np.pi*sigma**2) * np.exp(-0.5*(x-mean)**2/sigma**2)
    return np.log(gauss)

def FT_slope_labels(err,n):
    err_n="0.1"
    return ";~slope="+n+r"$\pm$"+err_n

def kurtosis(data):
    return cfp.round(st.kurtosis(data), 2, str_ret=True)

def func(x,a,n):
    return np.log(a*(x**n))

def func_gaussian(x,a,n):
    return np.log(a*(x**n))

def correction_bins(vmax_correc): # creates bins for the correction PDFs
    pos = list(np.logspace(0.1, vmax_correc, (vmax_correc/2)+1))
    neg = [ -x for x in list(reversed(pos))]
    tot = neg+pos
    return tot

def PDF_img_names_correc(i, sigma):
        sigma=cfp.round(sigma, 2, str_ret=True)
        correction_labels = [r"CO (1-0) at $1~t_\mathrm{ff}$", r"CO (1-0) at $1.2~t_\mathrm{ff}$", r"CO (2-1) at $1~t_\mathrm{ff}$", r"CO (2-1) at $1.2~t_\mathrm{ff}$"]
        return correction_labels[i]+r": $\sigma$ = "+sigma

def corrections(moment_map, ideal_moment_map, nth_moment):
    if nth_moment == 1:
        correction = moment_map - ideal_moment_map
        PDF_obj = cfp.get_pdf(correction, range=(-0.3, 0.3))
        sigma = np.std(correction)    
    else:
        correction = moment_map / ideal_moment_map
        PDF_obj = cfp.get_pdf(correction, range=(0.1, 10))
        sigma = np.std(correction)
    
    return correction, PDF_obj, sigma

def axes_format(tot_panels, coords_of_fig, xlabel, ylabel):
    coords_of_fig=str(coords_of_fig)
    rows=cols=int(np.sqrt(tot_panels))
    if coords_of_fig == '00' or '10': 
        axes_format=[None, ""]
        xlabel=None
        ylabel=ylabel
    elif coords_of_fig == '01' or '02' or '11' or '12':
        axes_format=["", ""]
        xlabel=None
        ylabel=None
    elif coords_of_fig == '21' or '22':
        axes_format=[None, ""]
        xlabel=xlabel
        ylabel=None
    else:
        axes_format=[None, None]
        xlabel=xlabel
        ylabel=ylabel

    return axes_format, xlabel, ylabel