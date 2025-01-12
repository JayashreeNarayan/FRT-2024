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
from util_scripts.all_functions import *
from util_scripts.universal_variables import *

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
                if key[:1]=='Z': self.zero_mom_ideal.append(resize_45(rescale_data(value), '2D'))
                elif key[:1]=='F': self.first_mom_ideal.append(resize_45(value, '2D'))
                elif key[:1]=='S': self.second_mom_ideal.append(resize_45(value, '2D'))
            else: # transpose all the other maps
                if key[:1]=='Z': self.zero_mom_ideal.append(rescale_data(value))
                elif key[:1]=='F': self.first_mom_ideal.append(value)
                elif key[:1]=='S': self.second_mom_ideal.append(value)
        print("all moments - idealsed made")
            
        # for the CO (1-0) case:
        for key, value in files_co10.items():
            if get_LOS(key) == 1: # have to resize the 45 degree maps             
                self.zero_mom_co10.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co10.append(resize_45(-first_moment(value, Vrange), '2D'))
                self.second_mom_co10.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co10.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co10.append(-first_moment(value, Vrange))
                self.second_mom_co10.append(second_moment(value, Vrange))
        print("all moments - co10 made")
        
        # for the CO (2-1) case:
        for key, value in files_co21.items():
            if get_LOS(key) == 1:             
                self.zero_mom_co21.append(rescale_data(resize_45(zero_moment(value, Vrange), '2D')))
                self.first_mom_co21.append(resize_45(-first_moment(value, Vrange), '2D'))
                self.second_mom_co21.append(resize_45(second_moment(value, Vrange), '2D'))
            else:    
                self.zero_mom_co21.append(rescale_data(zero_moment(value, Vrange)))
                self.first_mom_co21.append(-first_moment(value, Vrange))
                self.second_mom_co21.append(second_moment(value, Vrange))
        print("all moments - co21 made")
        
        # smoothing for the first moment
        self.first_mom_smooth_ideal=np.asarray([smoothing(i) for i in self.first_mom_ideal], dtype=object).astype(float)
        self.first_mom_smooth_co10=np.asarray([smoothing(i) for i in self.first_mom_co10], dtype=object).astype(float)
        self.first_mom_smooth_co21=np.asarray([smoothing(i) for i in self.first_mom_co21], dtype=object).astype(float)
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
        print("PDFs and sigmas obtained")

        # PDF objects and sigmas for isolated first moments
        self.first_mom_isolated_ideal_PDF_obj=  np.asarray([cfp.get_pdf(self.first_mom_isolated_ideal[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_ideal_sigma=    np.asarray([np.std(self.first_mom_isolated_ideal[i]) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co10_PDF_obj=   np.asarray([cfp.get_pdf(self.first_mom_isolated_co10[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co10_sigma=     np.asarray([np.std(self.first_mom_isolated_co10[i]) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co21_PDF_obj=   np.asarray([cfp.get_pdf(self.first_mom_isolated_co21[i], range=(-0.1, 0.1)) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        self.first_mom_isolated_co21_sigma=     np.asarray([np.std(self.first_mom_isolated_co21[i]) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
        print("PDFs and sigmas obtained from isolated first moment")

        # Kurtosis for the PDFs of first moment
        self.first_mom_ideal_kurt=np.asarray([kurtosis_own(self.first_mom_ideal[i].flatten()) for i in range(len(self.first_mom_ideal))], dtype=object)
        self.first_mom_co10_kurt=np.asarray([kurtosis_own(self.first_mom_co10[i].flatten()) for i in range(len(self.first_mom_co10))], dtype=object)
        self.first_mom_co21_kurt=np.asarray([kurtosis_own(self.first_mom_co21[i].flatten()) for i in range(len(self.first_mom_co21))], dtype=object)
        print("Kurtosis done")

        # Kurtosis for the PDFs of the first moment, after turb isolation
        self.first_mom_isolated_ideal = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_ideal]
        self.first_mom_isolated_co10 = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_co10]
        self.first_mom_isolated_co21 = [np.array(arr, dtype=float) for arr in self.first_mom_isolated_co21]
        self.first_mom_isolated_ideal_kurt=np.asarray([kurtosis_own(self.first_mom_isolated_ideal[i].flatten()) for i in range(len(self.first_mom_isolated_ideal))], dtype=object)
        self.first_mom_isolated_co10_kurt=np.asarray([kurtosis_own(self.first_mom_isolated_co10[i].flatten()) for i in range(len(self.first_mom_isolated_co10))], dtype=object)
        self.first_mom_isolated_co21_kurt=np.asarray([kurtosis_own(self.first_mom_isolated_co21[i].flatten()) for i in range(len(self.first_mom_isolated_co21))], dtype=object)
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
        #print(len(moments_.second_mom_ideal))
        self.zero_mom_ideal=[moments_.zero_mom_ideal[1], moments_.zero_mom_ideal[3]]
        self.zero_mom_co10=[moments_.zero_mom_co10[1], moments_.zero_mom_co10[3]]
        self.zero_mom_co21=[moments_.zero_mom_co21[1],moments_.zero_mom_co21[3]]

        self.first_mom_ideal=[moments_.first_mom_ideal[1], moments_.first_mom_ideal[3]]
        self.first_mom_co10=[moments_.first_mom_co10[1],moments_.first_mom_co10[3]]
        self.first_mom_co21=[moments_.first_mom_co21[1], moments_.first_mom_co21[3]]

        self.second_mom_ideal=[moments_.second_mom_ideal[1], moments_.second_mom_ideal[3]]
        self.second_mom_co10=[moments_.second_mom_co10[1],moments_.second_mom_co10[3]]
        self.second_mom_co21=[moments_.second_mom_co21[1], moments_.second_mom_co21[3]]

        # the smoothed first moment maps
        self.first_mom_smooth_ideal=[moments_.first_mom_smooth_ideal[1], moments_.first_mom_smooth_ideal[3]]
        self.first_mom_smooth_co10=[moments_.first_mom_smooth_co10[1], moments_.first_mom_smooth_co10[3]]
        self.first_mom_smooth_co21=[moments_.first_mom_smooth_co21[1], moments_.first_mom_smooth_co21[3]]

        # the isolated first moment maps
        self.first_mom_isolated_ideal=[moments_.first_mom_isolated_ideal[1], moments_.first_mom_isolated_ideal[3]]
        self.first_mom_isolated_co10=[moments_.first_mom_isolated_co10[1], moments_.first_mom_isolated_co10[3]]
        self.first_mom_isolated_co21=[moments_.first_mom_isolated_co21[1], moments_.first_mom_isolated_co21[3]]

        # PDF objects and sigmas for the first moment
        self.first_mom_ideal_PDF_obj=[moments_.first_mom_ideal_PDF_obj[1], moments_.first_mom_ideal_PDF_obj[3]]
        self.first_mom_co10_PDF_obj=[moments_.first_mom_co10_PDF_obj[1], moments_.first_mom_co10_PDF_obj[3]]
        self.first_mom_co21_PDF_obj=[moments_.first_mom_co21_PDF_obj[1], moments_.first_mom_co21_PDF_obj[3]]
        self.first_mom_ideal_sigma=[moments_.first_mom_ideal_sigma[1], moments_.first_mom_ideal_sigma[3]]
        self.first_mom_co10_sigma=[moments_.first_mom_co10_sigma[1], moments_.first_mom_co10_sigma[3]]
        self.first_mom_co21_sigma=[moments_.first_mom_co21_sigma[1], moments_.first_mom_co21_sigma[3]]

        # PDF objects and sigmas for the first moment after isolation
        self.first_mom_isolated_ideal_PDF_obj=[moments_.first_mom_isolated_ideal_PDF_obj[1], moments_.first_mom_isolated_ideal_PDF_obj[3]]
        self.first_mom_isolated_co10_PDF_obj=[moments_.first_mom_isolated_co10_PDF_obj[1], moments_.first_mom_isolated_co10_PDF_obj[3]]
        self.first_mom_isolated_co21_PDF_obj=[moments_.first_mom_isolated_co21_PDF_obj[1], moments_.first_mom_isolated_co21_PDF_obj[3]]
        self.first_mom_isolated_ideal_sigma=[moments_.first_mom_isolated_ideal_sigma[1], moments_.first_mom_isolated_ideal_sigma[3]]
        self.first_mom_isolated_co10_sigma=[moments_.first_mom_isolated_co10_sigma[1], moments_.first_mom_isolated_co10_sigma[3]]
        self.first_mom_isolated_co21_sigma=[moments_.first_mom_isolated_co21_sigma[1], moments_.first_mom_isolated_co21_sigma[3]]

        # kurtosis for the PDFs
        self.first_mom_ideal_kurt=[moments_.first_mom_ideal_kurt[1], moments_.first_mom_ideal_kurt[3]]
        self.first_mom_co10_kurt=[moments_.first_mom_co10_kurt[1], moments_.first_mom_co10_kurt[3]]
        self.first_mom_co21_kurt=[moments_.first_mom_co21_kurt[1], moments_.first_mom_co21_kurt[3]]

        # kurtosis for the PDFS after isolation
        self.first_mom_isolated_ideal_kurt=[moments_.first_mom_isolated_ideal_kurt[1], moments_.first_mom_isolated_ideal_kurt[3]]
        self.first_mom_isolated_co10_kurt=[moments_.first_mom_isolated_co10_kurt[1], moments_.first_mom_isolated_co10_kurt[3]]
        self.first_mom_isolated_co21_kurt=[moments_.first_mom_isolated_co21_kurt[1], moments_.first_mom_isolated_co21_kurt[3]]
        
        # FT objects 
        self.first_mom_ideal_FT_obj=[moments_.first_mom_ideal_FT_obj[1], moments_.first_mom_ideal_FT_obj[3]]
        self.first_mom_co10_FT_obj=[moments_.first_mom_co10_FT_obj[1], moments_.first_mom_co10_FT_obj[3]]
        self.first_mom_co21_FT_obj=[moments_.first_mom_co21_FT_obj[1], moments_.first_mom_co21_FT_obj[3]]

        # FT objects after isolation
        self.first_mom_isolated_ideal_FT_obj=[moments_.first_mom_isolated_ideal_FT_obj[1], moments_.first_mom_isolated_ideal_FT_obj[3]]
        self.first_mom_isolated_co10_FT_obj=[moments_.first_mom_isolated_co10_FT_obj[1], moments_.first_mom_isolated_co10_FT_obj[3]]
        self.first_mom_isolated_co21_FT_obj=[moments_.first_mom_isolated_co21_FT_obj[1], moments_.first_mom_isolated_co21_FT_obj[3]]

        # correction objects
        self.zero_mom_co10_correction_obj=[moments_.zero_mom_co10_correction_obj[1], moments_.zero_mom_co10_correction_obj[3]]
        self.zero_mom_co21_correction_obj=[moments_.zero_mom_co21_correction_obj[1], moments_.zero_mom_co21_correction_obj[3]]

        self.first_mom_co10_correction_obj=[moments_.first_mom_co10_correction_obj[1], moments_.first_mom_co10_correction_obj[3]]
        self.first_mom_co21_correction_obj=[moments_.first_mom_co21_correction_obj[1], moments_.first_mom_co21_correction_obj[3]]

        self.second_mom_co10_correction_obj=[moments_.second_mom_co10_correction_obj[1], moments_.second_mom_co10_correction_obj[3]]
        self.second_mom_co21_correction_obj=[moments_.second_mom_co21_correction_obj[1], moments_.second_mom_co21_correction_obj[3]]

        # correction obj for first mom after isolation
        self.first_mom_isolated_co10_correction_obj=[moments_.first_mom_isolated_co10_correction_obj[1], moments_.first_mom_isolated_co10_correction_obj[3]]
        self.first_mom_isolated_co21_correction_obj=[moments_.first_mom_isolated_co21_correction_obj[1], moments_.first_mom_isolated_co21_correction_obj[3]]
        print("Picking out the one for LOS = 45 degrees")

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

        self.PDF_xmin=xmin
        self.PDF_xmax=xmax
        self.PDF_ymin=ymin
        self.PDF_ymax=ymax

        self.correc_PDF_xmin = -0.3
        self.correc_PDF_xmax = 0.3
        self.correc_PDF_ymin = 0.1
        self.correc_PDF_ymax = 70

        self.kmin = 3
        self.kmax = 40
        self.FT_xpos = img_PDF_names_xpos+0.006
        self.FT_ypos = img_PDF_names_ypos-0.33
        self.params = {"a":[1e-4, 1e-2, 1], "n":[-4, -2, -1]}
    
    # To plot any moment map with colorbars
    def with_colorbar(self, data, text, file):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=self.xlabel, ylabel=self.ylabel, normalised_coords=True)     
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)              
        cfp.show_or_save_plot(save=file+'_cb.pdf')

    # To plot any moment map without colorbars
    def without_colorbar(self, data, text, file, coords_of_fig):
        axes_format, xlabel, ylabel = axes_format_func(coords_of_fig, self.xlabel, self.ylabel)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = False, cmap_label=self.cmap_label, axes_format=axes_format, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=xlabel, ylabel=ylabel, normalised_coords=True)    
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)               
        cfp.show_or_save_plot(save=file+'.pdf')
    
    def colorbar(self, panels):
        cfp.plot_colorbar(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, label=self.cmap_label, save=outpath+self.cmap+'_'+str(panels)+"_colorbar.pdf", panels=panels)
    
    # To plot general PDFs
    def PDF_panel(self, PDF_obj, sigma, file, text):
        xfit = np.linspace(self.PDF_xmin, self.PDF_xmax, 500)
        for i in range(len(PDF_obj)):
            cfp.plot(x=PDF_obj[i].bin_edges, y=PDF_obj[i].pdf, type='pdf', label=PDF_img_names(i, sigma[i]), color=line_colours[i])
            good_ind = PDF_obj[i].pdf > 0
            fitobj = cfp.fit(func=gauss_func, xdat=PDF_obj[i].bin_center[good_ind], ydat=np.log(PDF_obj[i].pdf[good_ind]), perr_method='statistical')
            cfp.plot(x=xfit, y=np.exp(gauss_func(xfit, *fitobj.popt)), alpha=0.5, color=line_colours[i], linestyle='dashed')
        cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos, text=text, backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
        cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[self.PDF_xmin, self.PDF_xmax], ylim=[self.PDF_ymin,self.PDF_ymax], legend_loc='upper left', save=file)
    
    # To plot the correction PDFs
    def PDF_corrections(self, PDF_obj, sigma, file, text):
        xfit = np.linspace(self.correc_PDF_xmin, self.correc_PDF_xmax, 500)
        for i in range(len(PDF_obj)):
            cfp.plot(x=PDF_obj[i].bin_edges, y=PDF_obj[i].pdf, type='pdf', label=PDF_img_names_correc(i, sigma[i]), color=line_colours[i])
            good_ind = PDF_obj[i].pdf > 0
            fitobj = cfp.fit(func=gauss_func, xdat=PDF_obj[i].bin_center[good_ind], ydat=np.log(PDF_obj[i].pdf[good_ind]), perr_method='statistical')
            cfp.plot(x=xfit, y=np.exp(gauss_func(xfit, *fitobj.popt)), alpha=0.5, color=line_colours[i], linestyle='dashed')
        cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos+0.08, text=text, backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
        cfp.plot(xlabel=correction_cmap_lables[1], ylabel="PDF", fontsize='small', ylog=True, xlim = [self.correc_PDF_xmin, self.correc_PDF_xmax], ylim=[self.correc_PDF_ymin, self.correc_PDF_ymax], legend_loc='upper left', save=file)
    
    # To plot all the power spectra graphs 
    def FT_panel(self, FTdata, file, text):
        for i in range(len(FTdata)):
            x=FTdata[i][0][1:self.kmax]; y=FTdata[i][1][1:self.kmax]
            params = self.params
            fit_values = cfp.fit(func, xdat=x[self.kmin:], ydat=np.log(list(y[self.kmin:])), perr_method='systematic', params=params)
            a=cfp.round(fit_values.popt[0], 2, str_ret=True); n=cfp.round(fit_values.popt[1], 2, str_ret=True)
            cfp.plot(x=x, y=y, label=img_names[i]+FT_slope_labels(fit_values.perr,n), color=line_colours[i])
            cfp.plot(x=x[self.kmin:], y=np.exp(func(x[self.kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle='dotted')
        secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
        secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
        secax1.tick_params(axis='x', direction='in', length=0, which = 'minor', top=False, bottom=True)
        cfp.plot(x=self.FT_xpos, y=self.FT_ypos, text=text, backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
        cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, save=file)
    
    # a function to plot the appendix figure 
    def without_colorbar_appendix(self, data, text, file, coords_of_fig,  xlabel_own, ylabel_own, angle):
        axes_format, xlabel, ylabel = axes_format_func(coords_of_fig, xlabel_own, ylabel_own)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = False, cmap_label=self.cmap_label, axes_format=axes_format, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=xlabel, ylabel=ylabel, normalised_coords=True)    
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)               
        ret.ax()[0].text(x=LOS_labels_xpos, y=LOS_labels_ypos, s=angle, color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)
        cfp.show_or_save_plot(save=file+'.pdf')
    
# class to plot zeroth moment maps
class zeroth_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[0]
        self.vmin=vmin_0
        self.vmax=vmax_0
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.cmap_label=cmap_labels[0]
    
    # Plotting moment map with colorbars for zero moment
    def with_colorbar(self, data, text, file):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=self.xlabel, ylabel=self.ylabel, normalised_coords=True)   
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)                
        cfp.show_or_save_plot(save=file+'_cb.pdf')

    # To plot any moment map without colorbars
    def without_colorbar(self, data, text, file, coords_of_fig):
        axes_format, xlabel, ylabel = axes_format_func(coords_of_fig, self.xlabel, self.ylabel)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = False, cmap_label=self.cmap_label, axes_format=axes_format, xlim=[-1,1], ylim=[-1,1], log=True, aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=xlabel, ylabel=ylabel, normalised_coords=True)    
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)               
        cfp.show_or_save_plot(save=file+'.pdf')
    
    def colorbar(self, panels):
        cfp.plot_colorbar(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, label=self.cmap_label, save=outpath+self.cmap+'_'+str(panels)+"_colorbar.pdf", log=True, panels=panels)
    

# class to plot second moment maps
class second_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[2]
        self.vmin=vmin_2
        self.vmax=vmax_2
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.cmap_label=cmap_labels[2]
    
    # Plotting moment map with colorbars for 2nd moment
    def with_colorbar(self, data, text, file):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = True, cmap_label=self.cmap_label, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=self.xlabel, ylabel=self.ylabel, normalised_coords=True)   
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)                
        cfp.show_or_save_plot(save=file+'_cb.pdf')
    
    # To plot any moment map without colorbars
    def without_colorbar(self, data, text, file, coords_of_fig):
        axes_format, xlabel, ylabel = axes_format_func(coords_of_fig, self.xlabel, self.ylabel)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, colorbar = False, cmap_label=self.cmap_label, axes_format=axes_format, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=xlabel, ylabel=ylabel, normalised_coords=True)    
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)               
        cfp.show_or_save_plot(save=file+'.pdf')
    
    def colorbar(self, panels):
        cfp.plot_colorbar(cmap=self.cmap, vmin=self.vmin, vmax=self.vmax, label=self.cmap_label, save=outpath+self.cmap+'_'+str(panels)+"_colorbar.pdf", panels=panels)

# class to plot correction maps
class correction_moment_plotter:
    def __init__(self):
        self.cmap=cmaps[3]
        self.vmin=[0.1, -0.3, 0.2]
        self.vmax=[10, 0.3, 5]
        self.xlabel=xyzlabels[1]
        self.ylabel=xyzlabels[3]
        self.log=[True, False, True]
        self.cmap_label=correction_cmap_lables
    
    # Plotting moment map with colorbars for correction maps
    def with_colorbar(self, data, text, file, moment):
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin[moment], vmax=self.vmax[moment], colorbar = True, cmap_label=self.cmap_label[moment], xlim=[-1,1], ylim=[-1,1], log=self.log[moment], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=self.xlabel, ylabel=self.ylabel, normalised_coords=True) 
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)         
        cfp.show_or_save_plot(save=file+'_cb.pdf')
    
    # To plot any moment map without colorbars
    def without_colorbar(self, data, text, file, moment, coords_of_fig):
        axes_format, xlabel, ylabel = axes_format_func(coords_of_fig, self.xlabel, self.ylabel)
        ret = cfp.plot_map(data, cmap=self.cmap, vmin=self.vmin[moment], vmax=self.vmax[moment], colorbar = False, cmap_label=self.cmap_label[moment], log=self.log[moment], axes_format=axes_format, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
        cfp.plot(ax=ret.ax()[0], xlabel=xlabel, ylabel=ylabel, normalised_coords=True)    
        ret.ax()[0].text(x=img_names_xpos, y=img_names_ypos, s=text , color='black', bbox=dict(facecolor='white', alpha=0.3, edgecolor='none'), transform=plt.gca().transAxes)               
        cfp.show_or_save_plot(save=file+'.pdf')
    
    def colorbar(self, panels, moment):
        cfp.plot_colorbar(cmap=self.cmap, vmin=self.vmin[moment], vmax=self.vmax[moment], label=self.cmap_label[moment], log=self.log[moment], save=outpath+self.cmap+'_'+str(panels)+"_moment_"+str(moment)+"_colorbar.pdf", panels=panels)
    
