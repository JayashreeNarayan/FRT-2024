import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *

from co_2_1 import co21
from co_1_0 import co10
from optically_thin import othin

ideal_1tff, ideal_SE = othin(correction='Y')
CO_21_1tff, CO_21_SE = co21(correction='Y')
CO_10_1tff, CO_10_SE = co10(correction='Y')

find_plots()

correction_CO_10_1tff = (CO_10_1tff[0]-ideal_1tff[0])/ideal_1tff[0] # 10 at 1tff
correction_CO_10_SE = (CO_10_SE[0]-ideal_SE[0])/ideal_SE[0] # 10 at SE

correction_CO_21_1tff = (CO_21_1tff[0]-ideal_1tff[0])/ideal_1tff[0] # 21 at 1tff
correction_CO_21_SE = (CO_21_SE[0]-ideal_SE[0])/ideal_SE[0] # 21 at SE

corrections = [correction_CO_10_1tff, correction_CO_10_SE, correction_CO_21_1tff, correction_CO_21_SE]

# Plotting the correction factor maps
cfp.plot_map(correction_CO_10_1tff, cmap=cmaps[3], colorbar=False, vmin=vmin_correc, vmax=vmax_correc, axes_format=["",None], symlog=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
t = plt.text(img_names_xpos, img_names_ypos, correction_labels[0] , transform=plt.gca().transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
cfp.plot(xlabel="", ylabel=xyzlabels[1], save=outpath+"correction_map_10_1tff.pdf")

cfp.plot_map(correction_CO_21_1tff, cmap=cmaps[3], colorbar=False ,vmin=vmin_correc, vmax=vmax_correc,  axes_format=["",""], symlog=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
t = plt.text(img_names_xpos, img_names_ypos, correction_labels[1] , transform=plt.gca().transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
cfp.plot(xlabel="", ylabel="", save=outpath+"correction_map_21_1tff.pdf")

cfp.plot_map(correction_CO_10_SE, cmap=cmaps[3], colorbar=False , vmin=vmin_correc, vmax=vmax_correc, axes_format=[None,None], symlog=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
t = plt.text(img_names_xpos, img_names_ypos, correction_labels[2] , transform=plt.gca().transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+"correction_map_10_SE.pdf")

cfp.plot_map(correction_CO_21_SE, cmap=cmaps[3], colorbar=False , vmin=vmin_correc, vmax=vmax_correc, axes_format=[None,""], symlog=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
t = plt.text(img_names_xpos, img_names_ypos, correction_labels[3] , transform=plt.gca().transAxes)
t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
cfp.plot(xlabel=xyzlabels[0], ylabel="", save=outpath+"correction_map_21_SE.pdf")

# Obtaining correction PDFs
for i in range(len(corrections)):
    K = cfp.get_pdf(corrections[i], bins=correction_bins_values) 
    PDF_correction_obj.append(K)
    correction_sigmas.append(np.std(corrections[i]))

# Plotting correction PDFs
for i in range(len(PDF_correction_obj)):
    if i==1 or i==3: alpha=0.5
    else: alpha=1
    if i==0 or i==2: line_color=line_colours[1]
    else: line_color=line_colours[2]
    cfp.plot(x=PDF_correction_obj[i].bin_edges, y=PDF_correction_obj[i].pdf, alpha=alpha, type='pdf', label=PDF_img_names_correc(i, correction_sigmas[i]), color=line_color)
cfp.plot(xlabel=correction_xlabel, ylabel="PDF", fontsize='small', ylog=True, xlim=[vmin_correc, vmax_correc], ylim=[ylim_min, ylim_max], legend_loc='upper left', save=outpath+"correction_PDF.pdf")
