#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os

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
    return mom1 / mom0

# Same as first_moment, but for the 2nd moment
def second_moment(PPV, Vrange):
    dv = Vrange[1]-Vrange[0] # get velocity channel width
    mom0 = zero_moment(PPV, Vrange) # moment-0 for normalisation
    mom2 = np.sum(PPV*Vrange**2, axis=2) * dv / mom0
    mom1 = np.sum(PPV*Vrange, axis=2) * dv / mom0
    mom2 = np.sqrt(mom2 - mom1**2)
    return mom2

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
    moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
    cmaps = ['plasma', 'seismic', 'viridis']
    cmap_labels = [r"Density (g/cm$^3$)", r"$v_z~\mathrm{(km\,s^{-1}})$", r"$\sigma_{v_z}~\mathrm{(km\,s^{-1}})$"]
    xyzlabels = ["$x$ (pc)", "$y$ (pc)", "$z$ (pc)"]

    # loop through chosen actions
    for action in args.action:

        # Plotting the optically thin first moment maps with cfpack and also smoothing them out and then obtaining the Gaussian-corrected maps
        if action == choices[0]:
            files = ["FMM_0.0_0.0.npy", "FMM_90.0_0.0.npy", "SMM_0.0_0.0.npy"]
            for file in files:
                data = np.load(path+"/Data_1tff/Othin/"+file)
                data = np.flipud(data).T
                if file == "SMM_0.0_0.0.npy": # Since the 2nd moment map needs different plot variables
                    vmin, vmax = get_vmin_vmax_centred(data) # Since this map is only needed for Fig 1., we don't have to use a universal vmin and vmax
                    cfp.plot_map(data, cmap=cmaps[2], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[2]) # colorbar needed since this is for Fig.1
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-3]+"pdf")
                else: 
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False) # Done for the 1st moment maps separately, needed for Fig. 2
                    ylabel = xyzlabels[1]
                    if file == "FMM_0.0_0.0.npy": # 0,0 is the XY map
                        xlabel = xyzlabels[0]
                    else: 
                        xlabel = xyzlabels[2] # 90,0 map is the XZ map
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                    # Also plotting the same with colorbars for Fig. 1
                    vmin, vmax = get_vmin_vmax_centred(data) # Since this is for Fig. 1, we cannot use a universal vmin and vmax
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[1])
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
                if file == "SMM_0.0_0.0.npy": # 2nd moment map is not to be smoothed or gaussian corrected 
                    continue
                else:
                    smooth_data = smoothing(data)
                    cfp.plot_map(smooth_data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False) # Needed for Fig. 2 so, no colorbar, universal vmin and vmax to be used.
                    ylabel = xyzlabels[1]
                    if file == "FMM_0.0_0.0.npy": 
                        xlabel = xyzlabels[0] 
                    else: 
                        xlabel = xyzlabels[2]
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_smooth.pdf")

                    # Gaussian-correction of the smoothed data
                    vmin = -0.4
                    vmax = +0.4
                    corrected_data_othin = data - smooth_data
                    cfp.plot_map(corrected_data_othin, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False)
                    ylabel=xyzlabels[1]
                    if file == "FMM_0.0_0.0.npy": xlabel=xyzlabels[0]
                    else: xlabel=xyzlabels[2]
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

        # Plotting the zeroth moment maps with flashplotlib directly from the FLASH data , used only in Fig. 1 so we need a colorbar
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -nolog -cmap plasma -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"Density (g/cm$^3$)\" -time_scale 0"
                cfp.run_shell_command(cmd)

        # PPV cubes - 0 moment map and consequently first moment map; smoothing and also gaussian correction
        if action == choices[2]:

            files = ["PPV_0_0.npy", "PPV_90_0.npy"]
            for file in files:
                moms = []

                # read PPV data and V axis
                PPV = np.load(path+"/Data_1tff/"+file)
                Vrange = np.load(path+"/Data_1tff/"+"Vrange.npy")

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange)
                    if imom==1: mom = first_moment(PPV, Vrange)
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps
                    
                    # plot moment maps, since PPV is used in both Fig.1 and 2, we need one set with colorbars and one set without
                    
                    # Set with a common colorbar, common colorbar for cmap=seismic is made at the end:                    
                    vmin, vmax = get_vmin_vmax_centred(moms[imom])
                    if imom==1: # only moment 1 has this universality so that Fig. 2 looks uniform
                        vmin=-0.4
                        vmax=+0.4
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False)
                    if file == "PPV_0_0.npy": # To plot the right labels
                        cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_map+".pdf")
                    else:
                        cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_map+".pdf")
                    
                    # Set with individual colorbars
                    vmin, vmax = get_vmin_vmax_centred(moms[imom]) # individual colorbars also means individual vmin and vmax
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, cmap_label=cmap_labels[imom])                    
                    if file == "PPV_0_0.npy": # To plot the right labels
                        cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")
                    else:
                        cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                # Make PDF of orginal mom1 and plot
                pdf_obj = cfp.get_pdf(moms[1])
                vmin, vmax = get_vmin_vmax_centred(moms[1])
                cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="pdf")
                cfp.plot(x=0.05, y=0.9, text="Low-pass-filtered moment 1, Optically thin case", backgroundcolor="white", fontsize="x-small", transform=plt.gca().transAxes)
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", ylog=True, xlim=[vmin,vmax])
                

                # Smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                vmin=-0.4 # Need universal vmin and vmax for Fig. 2, colorbar made at end
                vmax=+0.4
                cfp.plot_map(smooth_mom1, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False) # commmon colorbar for Fig. 2
                if file == "PPV_0_0.npy":
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")
                else:
                    cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")

                # Low-pass-filtered moment 1
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin, vmax=vmax, colorbar=False) # common colorbar for Fig. 2
                if file == "PPV_0_0.npy":
                    cfp.plot(xlabel=xyzlabels[0], ylabel=xyzlabels[1], save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")
                else:
                    cfp.plot(xlabel=xyzlabels[1], ylabel=xyzlabels[2], save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                # Make PDF and plot
                pdf_obj = cfp.get_pdf(corrected_data)
                cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="histogram")
                cfp.plot(x=0.05, y=0.9, text="Low-pass-filtered moment 1, Optically thick case", backgroundcolor="white", fontsize="x-small", transform=plt.gca().transAxes)
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", ylog=True, xlim=[vmin,vmax])

                # plotting a common colorbar, only for seismic, universal vmin and vmax
                cfp.plot_colorbar(cmap=cmaps[1], vmin=-0.4, vmax=+0.4, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar.pdf")
