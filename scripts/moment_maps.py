#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from astropy import constants as c

# function that gets theta from the file name and decides which direction is the LOS
def get_LOS(file):
    theta = file[4:8]
    if theta == "00.0": return 0 # the no.s are based on LOS_labels
    if theta == "22.5": return 1
    if theta == "45.0": return 2
    if theta == "67.5": return 3
    if theta == "90.0": return 4

# function that divides data by its mean to avoid units
def rescale_data(data):
    return data/np.mean(data)

# function that creates the 0-moment map from a PPV cube with the velocity axis=2 and velocity channels in 'Vrange'
def zero_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = np.sum(PPV, axis=2) * dv
    return mom0.T

# Same as zero_moment, but for the 1st moment
def first_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = zero_moment(PPV, Vrange) # moment-0 for normalisation
    mom1 = np.sum(PPV*Vrange, axis=2) * dv
    return (mom1 / mom0).T

# Same as first_moment, but for the 2nd moment
def second_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = zero_moment(PPV, Vrange) # moment-0 for normalisation
    mom2 = np.sum(PPV*Vrange**2, axis=2) * dv / mom0
    mom1 = np.sum(PPV*Vrange, axis=2) * dv / mom0
    mom2 = np.sqrt(mom2 - mom1**2)
    return mom2.T

# function that returns a Gaussian-smoothed version of the data - data being a 2D array
def smoothing(data):
    npix = data.shape[0]
    return cfp.gauss_smooth(data, fwhm=npix/2, mode='nearest')

def get_vmin_vmax_centred(data):
    minmax = np.max([-data.min(), data.max()])
    return -minmax, +minmax

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot files')

    choices = ['othin', 'flash', 'ppv']
    parser.add_argument('-a', '--action', metavar='action', nargs='*', default=choices, choices=choices,
                        help='Choice: Between plotting the first moment maps with flashplotlib directly from the FLASH data (flash), plotting the optically thin first moment maps with cfpack (othin) and plotting the optically thick moment maps from PPV cubes (ppv)')

    args = parser.parse_args()

    # set paths and create plot directory
    path = "../Data/"
    outpath = "../plots/"
    if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

    # set some global option/variables
    vmin = -0.4
    vmax = +0.4
    xmin=-0.6
    xmax=+0.6
    moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
    cmaps = ['plasma', 'seismic', 'viridis']
    cmap_labels = [r"I/$\langle I \rangle$", r"$v_{LOS}~\mathrm{(km\,s^{-1}})$", r"$\sigma_{v_{LOS}}~\mathrm{(km\,s^{-1}})$"]
    LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 0.383 \\ 0 \\ 0.924 \end{array}\right) $", r"$\left(\begin{array}{c} 0.707 \\ 0 \\ 0.707 \end{array}\right) $", r"$\left(\begin{array}{c} 0.924 \\ 0 \\ 0.383 \end{array}\right) $" , r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
    xyzlabels = ["POS", "POS", "POS"]
    img_names = ["Synthetic emission"+"\n"+"CO (1-0)", "Optically thin emission"]

    # loop through chosen actions
    for action in args.action:

        # Plotting the optically thin first moment maps with cfpack and also smoothing them out and then obtaining the Gaussian-corrected maps
        if action == choices[0]:
            files = ["FMM_00.0_0.0.npy", "FMM_22.5_0.0.npy", "FMM_45.0_0.0.npy", "FMM_67.5_0.0.npy", "FMM_90.0_0.0.npy", "SMM_00.0_0.0.npy", "SMM_22.5_0.0.npy", "SMM_45.0_0.0.npy", "SMM_67.5_0.0.npy", "SMM_90.0_0.0.npy", "ZMM_00.0_0.0.npy", "ZMM_22.5_0.0.npy", "ZMM_45.0_0.0.npy", "ZMM_67.5_0.0.npy", "ZMM_90.0_0.0.npy"]
            for file in files:
                data = np.load(path+"/Data_1tff/Othin/"+file)
                data = data.T
                if file[:1] == "S": # Since the 2nd moment map needs different plot variables
                    vmin = 0.
                    vmax = 1.
                    if get_LOS(file) == 0:
                        cfp.plot_map(data, cmap=cmaps[2], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect='equal') # colorbar needed since this is for Fig.1
                        t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes) # for img_name
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-3]+"pdf")

                elif file == "ZMM_00.0_0.0.npy": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                    data = rescale_data(data)
                    vmin = 0.
                    vmax = 7.
                    cfp.plot_map(data, cmap=cmaps[0], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect='equal') # colorbar needed since this is for Fig.1
                    t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-3]+"pdf")

                else:
                    vmin = -0.4
                    vmax = +0.4
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect='equal') # Done for the 1st moment maps separately, needed for Fig. 2
                    ylabel = xyzlabels[1]
                    if file == "FMM_00.0_0.0.npy": # 0,0 is the XY map
                        xlabel = xyzlabels[0]
                    else:
                        xlabel = xyzlabels[2] # 90,0 map is the XZ map
                    t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                    # Also plotting the same with colorbars for Fig. 1
                    vmin = -0.4
                    vmax = +0.4
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect='equal')
                    t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                    # Creating the PDF for the optically thin case - 1st-moment, without filtering
                    pdf_obj = cfp.get_pdf(data)
                    sigma = round(np.std(data),3)
                    cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="pdf")
                    cfp.plot(x=0.05, y=0.75, text=r"1st-moment"+"\n"+img_names[1]+r"; $\sigma$ = "+str(sigma)+r"$~\mathrm{km\,s^{-1}}$", axes_format=["",None], fontsize='small', backgroundcolor="white", transform=plt.gca().transAxes)
                    t = plt.text(0.85, 0.8, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0, linewidth=0))
                    cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin,xmax], ylim=[1.e-2,5.e1])

                # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
                if file == "FMM_00.0_0.0.npy": # 2nd and 0th moment map is not to be smoothed or gaussian corrected 
                    smooth_data = smoothing(data)
                    vmin = -0.4
                    vmax = +0.4
                    cfp.plot_map(smooth_data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["",""], xlim=[-1,1], ylim=[-1,1], aspect='equal') # Needed for Fig. 2 so, no colorbar, universal vmin and vmax to be used.
                    ylabel = xyzlabels[1]
                    if file == "FMM_00.0_0.0.npy":
                        xlabel = xyzlabels[0]
                    else:
                        xlabel = xyzlabels[2]
                    t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_smooth.pdf")

                    # Gaussian-correction of the smoothed data
                    vmin = -0.4
                    vmax = +0.4
                    corrected_data_othin = data - smooth_data
                    cfp.plot_map(corrected_data_othin, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["",""], xlim=[-1,1], ylim=[-1,1], aspect='equal')
                    ylabel=xyzlabels[1]
                    if file == "FMM_00.0_0.0.npy":
                        xlabel=xyzlabels[0]
                    else:
                        xlabel=xyzlabels[2]
                    t = plt.text(0.03, 0.93, img_names[1] , transform=plt.gca().transAxes) 
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                    # PDF of the low-pass-filtered data - 1st-moment
                    pdf_obj = cfp.get_pdf(corrected_data_othin)
                    cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="histogram", axes_format=["",""])
                    sigma = round(np.std(corrected_data_othin),3)
                    cfp.plot(x=0.05, y=0.75, text=r"Low-pass-filtered 1st-moment"+"\n"+img_names[1]+r"; $\sigma$ = "+str(sigma)+r"$~\mathrm{km\,s^{-1}}$", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
                    t = plt.text(0.85, 0.8, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0., linewidth=0))
                    cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[1.e-2, 5.e1])

        # Plotting the zeroth moment maps with flashplotlib directly from the FLASH data , used only in Fig. 1 so we need a colorbar
        '''
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -d vel"+dir+" -nolog -cmap seismic -mw -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"velocity\" -time_scale 0"
                #cmd = "flashplotlib.py -i "+file+" -nolog -cmap plasma -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"Density (g/cm$^3$)\" -time_scale 0"
                cfp.run_shell_command(cmd)
        '''
        # PPV cubes - 0 moment map and consequently first moment map; smoothing and also gaussian correction
        if action == choices[2]:

            files = ["PPV_0_0.npy"] 
            for file in files:
                moms = []

                # read PPV data and V axis
                PPV = np.load(path+"/Data_1tff/"+file) # loading the data
                PPV = rescale_data(PPV) # rescaling data to data/mean(data)
                Vrange = np.load(path+"/Data_1tff/"+"Vrange.npy")

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange)
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the optically thin images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and 2, we need one set with colorbars and one set without
                    # Set with a common colorbar (Fig. 2)        
                    vmin = 0.
                    if imom == 0:
                        vmax=7.
                    if imom == 1: # only moment 1 has this universality 
                        vmin=-0.4
                        vmax=+0.4
                    if imom == 2:
                        vmax=1.
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=[None, None], xlim=[-1,1], ylim=[-1,1], aspect='equal')
                    if file == "PPV_0_0.npy": # To plot the right labels
                        t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))                        
                        cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_map+".pdf")
                    else:
                        t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_map+".pdf")

                    # Set with individual colorbars (Fig. 1)
                    vmin = 0.
                    if imom == 0:
                        vmax=7.
                    if imom == 1: # only moment 1 has this universality 
                        vmin=-0.4
                        vmax=+0.4
                    if imom == 2:
                        vmax=1.
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[imom], xlim=[-1,1], ylim=[-1,1], aspect='equal')
                    if file == "PPV_0_0.npy": # To plot the right labels
                        t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")
                    else:
                        t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                # plotting a common colorbar, only for seismic, universal vmin and vmax
                cfp.plot_colorbar(cmap=cmaps[1], vmin=-0.4, vmax=+0.4, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar.pdf", panels=2)

                # Make PDF of orginal mom1 and plot
                pdf_obj = cfp.get_pdf(moms[1])
                cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="pdf")
                sigma = round(np.std(moms[1]),3)
                cfp.plot(x=0.05, y=0.7, text=r"1st-moment"+"\n"+img_names[0]+r"; $\sigma$ = "+str(sigma)+r"$~\mathrm{km\,s^{-1}}$",  fontsize='small', backgroundcolor="white", transform=plt.gca().transAxes)
                t = plt.text(0.85, 0.8, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0., linewidth=0))
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", axes_format=[None,None], fontsize='small', ylog=True, xlim=[xmin,xmax], ylim=[1.e-2,5.e1])

                # Smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                vmin=-0.4 # Need a universal vmin and vmax for Fig. 2, colorbar made at end
                vmax=+0.4
                cfp.plot_map(smooth_mom1, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False, axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect='equal') # commmon colorbar for Fig. 2
                t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if file == "PPV_0_0.npy":
                    cfp.plot(xlabel=xyzlabels[0], ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")
                else:
                    cfp.plot(xlabel=xyzlabels[1], ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")

                # Generating corrected map and then plotting it
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                vmin=-0.4 # Need a universal vmin and vmax for Fig. 2, colorbar made at end
                vmax=+0.4
                cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect ='equal') # common colorbar for Fig. 2
                t = plt.text(0.03, 0.85, img_names[0] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                t = plt.text(0.8, 0.85, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if file == "PPV_0_0.npy":
                    cfp.plot(xlabel=xyzlabels[0], ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")
                else:
                    cfp.plot(xlabel=xyzlabels[1], ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                # Make PDF of low-pass-filtered moment 1 and also plot it
                pdf_obj = cfp.get_pdf(corrected_data)
                cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="histogram")
                sigma = round(np.std(corrected_data),3)
                cfp.plot(x=0.05, y=0.7, text=r"Low-pass-filtered 1st-moment"+"\n"+img_names[0]+r"; $\sigma$ = "+str(sigma)+r"$~\mathrm{km\,s^{-1}}$", axes_format=[None,""], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
                t = plt.text(0.85, 0.8, LOS_labels[2] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0., linewidth=0))
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[1.e-2,5.e1])
