#!/usr/bin/env python
# This file is the classes of all the data

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *
from universal_variables import *

# all the arrays or lists in this class are in the order of angles: 0, 45, 90, 45_SE
class moments:
    def __init__(self):
        # getting the moments
        self.zero_mom_ideal=[]
        self.zero_mom_co10=[]
        self.zero_mom_co21=[]

        self.first_mom_ideal=[]
        self.first_mom_co10=[]
        self.first_mom_co21=[]

        self.second_mom_ideal=[]
        self.second_mom_co10=[]
        self.second_mom_co21=[]

        # for the Ideal case
        for key, value in files_ideal.items():
            if get_LOS(key) == 1: # have to resize the 45 degree maps   
                if key[:31]=='Z': self.zero_mom_ideal.append(resize_45(rescale_data(value), '2D'))
                elif key[:1]=='F': self.first_mom_ideal.append(resize_45(value, '2D'))
                elif key[:1]=='S': self.second_mom_ideal.append(resize_45(value, '2D'))
            else:
                if key[:1]=='Z': self.zero_mom_ideal.append(rescale_data(value))
                elif key[:1]=='F': self.first_mom_ideal.append(value)
                elif key[:]=='S': self.second_mom_ideal.append(value)
        print("all moments - idealsed made")
            
        # for the CO (1-0) case:
        for key, value in files_co10.items():
            if get_LOS(key) == 1: # have to resize the 45 degree maps             
                self.zero_mom_co10.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co10.append(resize_45(first_moment(value, Vrange), '2D'))
                self.second_mom_co10.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co10.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co10.append(first_moment(value, Vrange))
                self.second_mom_co10.append(second_moment(value, Vrange))
        print("all moments - co10 made")
        
        # for the CO (2-1) case:
        for key, value in files_co21.items():
            if get_LOS(key) == 1:             
                self.zero_mom_co21.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co21.append(resize_45(first_moment(value, Vrange), '2D'))
                self.second_mom_co21.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co21.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co21.append(first_moment(value, Vrange))
                self.second_mom_co21.append(second_moment(value, Vrange))
        print("all moments - co21 made")
        
        # smoothing for the first moment
        self.first_mom_smooth_ideal=np.asarray([smoothing(i) for i in self.first_mom_ideal], dtype=object)
        self.first_mom_smooth_co10=np.asarray([smoothing(i) for i in self.first_mom_co10], dtype=object)
        self.first_mom_smooth_co21=np.asarray([smoothing(i) for i in self.first_mom_co21], dtype=object)
        print("smoothed first moment")

        # isolation for the first moment
        self.first_mom_isolated_ideal=np.asarray([self.first_mom_ideal[i] - self.first_mom_smooth_ideal[i] for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_isolated_co10=np.asarray([self.first_mom_co10[i] - self.first_mom_smooth_co10[i] for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_isolated_co21=np.asarray([self.first_mom_co10[i] - self.first_mom_smooth_co21[i] for i in range(len(self.first_mom_co21))], dtype=object)
        print("isolated turb from first moment")

        # PDF objects and sigmas for the first moments
        self.first_mom_ideal_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_ideal[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_ideal_sigma=np.asarray([np.std(self.first_mom_ideal[i]) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_co10_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_co10[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_co10_sigma=np.asarray([np.std(self.first_mom_co10[i]) for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_co21_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_co21[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_co21))], dtype=object)
        self.first_mom_co21_sigma=np.asarray([np.std(self.first_mom_co21[i]) for i in range(len(self.first_mom_co21))], dtype=object)
        print(r"PDFs and $\sigma$s obtained")

        # PDF objects and sigmas for isolated first moments
        self.first_mom_isolated_ideal_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_isolated_ideal[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_ideal_sigma=np.asarray([np.std(self.first_mom_isolated_ideal[i]) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co10_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_isolated_co10[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co10_sigma=np.asarray([np.std(self.first_mom_isolated_co10[i]) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co21_PDF_obj=np.asarray([cfp.get_pdf(self.first_mom_isolated_co21[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        self.first_mom_isolated_co21_sigma=np.asarray([np.std(self.first_mom_isolated_co21[i]) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        print(r"PDFs and $\sigma$s obtained from isolated first moment")

        # Kurtosis for the PDFs of first moment
        self.first_mom_ideal_kurt=np.asarray([kurtosis(self.first_mom_ideal[i].flatten()) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_co10_kurt=np.asarray([kurtosis(self.first_mom_co10[i].flatten()) for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_co21_kurt=np.asarray([kurtosis(self.first_mom_co21[i].flatten()) for i in range(len(self.first_mom_co21))], dtype=object)
        print("Kurtosis done")

        # Kurtosis for the PDFs of the first moment, after turb isolation
        self.first_mom_isolated_ideal = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_ideal]
        self.first_mom_isolated_co10 = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_co10]
        self.first_mom_isolated_co21 = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_co21]
        self.first_mom_isolated_ideal_kurt=np.asarray([kurtosis(self.first_mom_isolated_ideal[i].flatten()) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co10_kurt=np.asarray([kurtosis(self.first_mom_isolated_co10[i].flatten()) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co21_kurt=np.asarray([kurtosis(self.first_mom_isolated_co21[i].flatten()) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        print("Kurtosis for isolated done")
        
        # FT objects for the first moments
        self.first_mom_ideal_FT_obj=np.asarray([fourier_spectrum(self.first_mom_ideal[i]) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_co10_FT_obj=np.asarray([fourier_spectrum(self.first_mom_co10[i]) for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_co21_FT_obj=np.asarray([fourier_spectrum(self.first_mom_co21[i]) for i in range(len(self.first_mom_co21))], dtype=object)
        print("FT obj done")

        # FT objects for isolated first moments
        self.first_mom_isolated_ideal_FT_obj=np.asarray([fourier_spectrum(self.first_mom_isolated_ideal[i]) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co10_FT_obj=np.asarray([fourier_spectrum(self.first_mom_isolated_co10[i]) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co21_FT_obj=np.asarray([fourier_spectrum(self.first_mom_isolated_co21[i]) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        print("FT obj for isolated done")

        # correction maps and correction PDFs for all moments
        self.zero_mom_co10_correction_obj=np.asarray([corrections(self.zero_mom_co10[i], self.zero_mom_ideal[i], 0) for i in range(len(self.zero_mom_ideal))], dtype=object)
        self.zero_mom_co21_correction_obj=np.asarray([corrections(self.zero_mom_co21[i], self.zero_mom_ideal[i], 0) for i in range(len(self.zero_mom_ideal))], dtype=object)

        self.first_mom_co10_correction_obj=np.asarray([corrections(self.first_mom_co10[i], self.first_mom_ideal[i], 1) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_co21_correction_obj=np.asarray([corrections(self.first_mom_co21[i], self.first_mom_ideal[i], 1) for i in range(len(self.first_mom_ideal))], dtype=object)

        self.second_mom_co10_correction_obj=np.asarray([corrections(self.second_mom_co10[i], self.second_mom_ideal[i], 2) for i in range(len(self.second_mom_ideal))], dtype=object)
        self.second_mom_co21_correction_obj=np.asarray([corrections(self.second_mom_co21[i], self.second_mom_ideal[i], 2) for i in range(len(self.second_mom_ideal))], dtype=object)
        print("Correction maps done")

        # correction maps and correction PDFs for first moment after turbulence isolation
        self.first_mom_isolated_co10_correction_obj=np.asarray([corrections(self.first_mom_isolated_co10[i], self.first_mom_isolated_ideal[i], 1) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co21_correction_obj=np.asarray([corrections(self.first_mom_isolated_co21[i], self.first_mom_isolated_ideal[i], 1) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        print("Correction maps after isolation done")

#moment_1 = moments()

# class to isolate the 45 degree maps at 1tff and SE
class moments_45:
    def __init__(self):
        moments_ = moments() # make the class with all the variables required across all angles
        
        # the moment maps
        self.zero_mom_ideal=[moments_.zero_mom_ideal[1], moments_.zero_mom_ideal[4]]
        self.zero_mom_co10=[moments_.zero_mom_co10[1], moments_.zero_mom_co10[4]]
        self.zero_mom_co21=[moments_.zero_mom_co21[1],moments_.zero_mom_co21[4]]

        self.first_mom_ideal=[moments_.first_mom_ideal[1], moments_.first_mom_ideal[4]]
        self.first_mom_co10=[moments_.first_mom_co10[1],moments_.first_mom_co10[4]]
        self.first_mom_co21=[moments_.first_mom_co21[1], moments_.first_mom_co21[4]]

        self.second_mom_ideal=[moments_.second_mom_ideal[1], moments_.second_mom_ideal[4]]
        self.second_mom_co10=[moments_.first_mom_co10[1],moments_.first_mom_co10[4]]
        self.second_mom_co21=[moments_.second_mom_co21[1], moments_.second_mom_co21[4]]

        # the smoothed first moment maps
        self.first_mom_smooth_ideal=[moments_.first_mom_smooth_ideal[1], moments_.first_mom_smooth_ideal[4]]
        self.first_mom_smooth_co10=[moments_.first_mom_smooth_co10[1], moments_.first_mom_smooth_co10[4]]
        self.first_mom_smooth_co21=[moments_.first_mom_smooth_co21[1], moments_.first_mom_smooth_co21[4]]

        # the isolated first moment maps
        self.first_mom_isolated_ideal=[moments_.first_mom_isolated_ideal[1], moments_.first_mom_isolated_ideal[4]]
        self.first_mom_isolated_co10=[moments_.first_mom_isolated_co10[1], moments_.first_mom_isolated_co10[4]]
        self.first_mom_isolated_co21=[moments_.first_mom_isolated_co21[1], moments_.first_mom_isolated_co21[4]]

        # PDF objects and sigmas for the first moment
        self.first_mom_ideal_PDF_obj=[moments_.first_mom_ideal_PDF_obj[1], moments_.first_mom_ideal_PDF_obj[4]]
        self.first_mom_co10_PDF_obj=[moments_.first_mom_co10_PDF_obj[1], moments_.first_mom_co10_PDF_obj[4]]
        self.first_mom_co21_PDF_obj=[moments_.first_mom_co21_PDF_obj[1], moments_.first_mom_co21_PDF_obj[4]]
        self.first_mom_ideal_sigma=[moments_.first_mom_ideal_sigma[1], moments_.first_mom_ideal_sigma[4]]
        self.first_mom_co10_sigma=[moments_.first_mom_co10_sigma[1], moments_.first_mom_co10_sigma[4]]
        self.first_mom_co21_sigma=[moments_.first_mom_co21_sigma[1], moments_.first_mom_co21_sigma[4]]

        # PDF objects and sigmas for the first moment after isolation
        self.first_mom_isolated_ideal_PDF_obj=[moments_.first_mom_isolated_ideal_PDF_obj[1], moments_.first_mom_isolated_ideal_PDF_obj[4]]
        self.first_mom_isolated_co10_PDF_obj=[moments_.first_mom_isolated_co10_PDF_obj[1], moments_.first_mom_isolated_co10_PDF_obj[4]]
        self.first_mom_isolated_co21_PDF_obj=[moments_.first_mom_isolated_co21_PDF_obj[1], moments_.first_mom_isolated_co21_PDF_obj[4]]
        self.first_mom_isolated_ideal_sigma=[moments_.first_mom_isolated_ideal_sigma[1], moments_.first_mom_isolated_ideal_sigma[4]]
        self.first_mom_isolated_co10_sigma=[moments_.first_mom_isolated_co10_sigma[1], moments_.first_mom_isolated_co10_sigma[4]]
        self.first_mom_isolated_co21_sigma=[moments_.first_mom_isolated_co21_sigma[1], moments_.first_mom_isolated_co21_sigma[4]]

        # kurtosis for the PDFs
        self.first_mom_ideal_kurt=[moments_.first_mom_ideal_kurt[1], moments_.first_mom_ideal_kurt[4]]
        self.first_mom_co10_kurt=[moments_.first_mom_co10_kurt[1], moments_.first_mom_co10_kurt[4]]
        self.first_mom_co21_kurt=[moments_.first_mom_co21_kurt[1], moments_.first_mom_co21_kurt[4]]

        # kurtosis for the PDFS after isolation
        self.first_mom_isolated_ideal_kurt=[moments_.first_mom_isolated_ideal_kurt[1], moments_.first_mom_isolated_ideal_kurt[4]]
        self.first_mom_isolated_co10_kurt=[moments_.first_mom_isolated_co10_kurt[1], moments_.first_mom_isolated_co10_kurt[4]]
        self.first_mom_isolated_co21_kurt=[moments_.first_mom_isolated_co21_kurt[1], moments_.first_mom_isolated_co21_kurt[4]]
        
        # FT objects 
        self.first_mom_ideal_FT_obj=[moments_.first_mom_ideal_FT_obj[1], moments_.first_mom_ideal_FT_obj[4]]
        self.first_mom_co10_FT_obj=[moments_.first_mom_co10_FT_obj[1], moments_.first_mom_co10_FT_obj[4]]
        self.first_mom_co21_FT_obj=[moments_.first_mom_co21_FT_obj[1], moments_.first_mom_co21_FT_obj[4]]

        # FT objects after isolation
        self.first_mom_isolated_ideal_FT_obj=[moments_.first_mom_isolated_ideal_FT_obj[1], moments_.first_mom_isolated_ideal_FT_obj[4]]
        self.first_mom_isolated_co10_FT_obj=[moments_.first_mom_isolated_co10_FT_obj[1], moments_.first_mom_isolated_co10_FT_obj[4]]
        self.first_mom_isolated_co21_FT_obj=[moments_.first_mom_isolated_co21_FT_obj[1], moments_.first_mom_isolated_co21_FT_obj[4]]

        # correction objects
        self.zero_mom_co10_correction_obj=[moments_.zero_mom_co10_correction_obj[1], moments_.zero_mom_co10_correction_obj[4]]
        self.zero_mom_co21_correction_obj=[moments_.zero_mom_co21_correction_obj[1], moments_.zero_mom_co21_correction_obj[4]]

        self.first_mom_co10_correction_obj=[moments_.first_mom_co10_correction_obj[1], moments_.first_mom_co10_correction_obj[4]]
        self.first_mom_co21_correction_obj=[moments_.first_mom_co21_correction_obj[1], moments_.first_mom_co21_correction_obj[4]]

        self.second_mom_co10_correction_obj=[moments_.second_mom_co10_correction_obj[1], moments_.second_mom_co10_correction_obj[4]]
        self.second_mom_co21_correction_obj=[moments_.second_mom_co21_correction_obj[1], moments_.second_mom_co21_correction_obj[4]]

        # correction obj for first mom after isolation
        self.first_mom_isolated_co10_correction_obj=[moments_.first_mom_isolated_co10_correction_obj[1], moments_.first_mom_isolated_co10_correction_obj[4]]
        self.first_mom_isolated_co21_correction_obj=[moments_.first_mom_isolated_co21_correction_obj[1], moments_.first_mom_isolated_co21_correction_obj[4]]
        print(r"Picking out the one for LOS = 45$^circ$")

#moment_45 = moments_45()

# class to plot first moment maps
class first_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[1]
        self.vmin=vmin_1
        self.vmax=vmax_1
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.cmap_label=cmap_labels[1]
    
    def with_colorbar(self, data, text, file):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label,xlabel=self.xlabel, ylabel=self.ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=text, normalised_coords=True)          
        cfp.show_or_save_plot(save=file+'_cb.pdf')
        print("saved at "+file+'_cb.pdf')

    def without_colorbar(self, data, text, save, coords_of_fig, tot_panels):
        axes_format, xlabel, ylabel = axes_format(tot_panels, coords_of_fig, self.xlabel, self.ylabel)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = False, cmap_label=self.cmap_label,xlabel=xlabel, ylabel=ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=text, normalised_coords=True)          
        cfp.show_or_save_plot(save=save+'.pdf')
        print("img saved at "+save+'.pdf')
        cfp.plot_colorbar(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, label=self.cmap_label, save=outpath+self.cmap+"_colorbar.pdf", panels=tot_panels)
        print("colorbar saved at "+outpath+self.cmap+"_colorbar.pdf")

# class to plot zeroth moment maps
class zeroth_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[0]
        self.vmin=vmin_0
        self.vmax=vmax_0
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.cmap_label=cmap_labels[0]
    
    def with_colorbar(self, data, text, save):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label,xlabel=self.xlabel, ylabel=self.ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=text, normalised_coords=True)          
        cfp.show_or_save_plot(save=save+'_cb.pdf')

# class to plot second moment maps
class second_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[2]
        self.vmin=vmin_2
        self.vmax=vmax_2
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.cmap_label=cmap_labels[2]
    
    def with_colorbar(self, data, text, save):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label,xlabel=self.xlabel, ylabel=self.ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=text, normalised_coords=True)          
        cfp.show_or_save_plot(save=save+'_cb.pdf')


        