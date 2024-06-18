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
    cmap_labels = [r"Density (g/cm$^3$)", r"$v$ (km/s)", r"$\sigma_v$ (km/s)"]

    # loop through chosen actions
    for action in args.action:

        # Plotting the optically thin first moment maps with cfpack
        if action == choices[0]:
            files = ["FMM_0.0_0.0.npy", "FMM_90.0_0.0.npy"]
            for file in files:
                data = np.load(path+"Data_1tff/Othin/"+file)
                data = np.flipud(data).T
                cfp.plot_map(data, cmap='seismic',cmap_label=r"$v$ (km/s)", vmin=vmin, vmax=vmax, save=outpath+file[:-3]+"pdf")
                smooth_data=smoothing(data)
                cfp.plot_map(smooth_data,cmap='seismic',cmap_label=r"$v$ (km/s)",save=outpath+file[:-4]+"_smooth.pdf")

        # Plotting the first moment maps with flashplotlib directly from the FLASH data
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -d vel"+dir+" -nolog -cmap seismic -mw -direction "+dir+" -outtype pdf -outdir "+outpath+" -vmin "+str(vmin*1e5)+" -vmax "+str(vmax*1e5)
                cfp.run_shell_command(cmd)

        # PPV cubes - 0 moment map and consequently first moment map; smoothing and also gaussian correction
        if action == choices[2]:

            files = ["PPV_0_0.npy", "PPV_90_0.npy"]
            for file in files:
                moms = []

                # read PPV data and V axis
                PPV = np.load(path+"Data_1tff/"+file)
                Vrange = np.load(path+"Data_1tff/"+"Vrange.npy")

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    vmin = None; vmax = None
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange)
                    if imom==1:
                        mom = first_moment(PPV, Vrange)
                        vmin, vmax = get_vmin_vmax_centred(mom)
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps
                    # plot moment maps
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], cmap_label=cmap_labels[imom], save=outpath+file[:-4]+"_"+moment_map+".pdf", vmin=vmin, vmax=vmax)

                # make PDF of orginal mom1 and plot
                pdf_obj = cfp.get_pdf(moms[1])
                vmin, vmax = get_vmin_vmax_centred(moms[1])
                cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="pdf", save=outpath+file[:-4]+"_"+moment_maps[1]+"_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", ylog=True, xlim=[vmin,vmax])

                # smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                vmin, vmax = get_vmin_vmax_centred(smooth_mom1)
                cfp.plot_map(smooth_mom1, cmap=cmaps[1], cmap_label=cmap_labels[1], save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf", vmin=vmin, vmax=vmax)

                # Low-pass-filtered moment 1
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                vmin, vmax = get_vmin_vmax_centred(corrected_data)
                cfp.plot_map(corrected_data, cmap=cmaps[1], cmap_label=cmap_labels[1], save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf", vmin=vmin, vmax=vmax)
                # make PDF and plot
                pdf_obj = cfp.get_pdf(corrected_data)
                plt = cfp.plot(x=pdf_obj.bin_edges, y=pdf_obj.pdf, type="histogram")
                cfp.plot(x=0.1, y=0.9, text="Low-pass-filtered moment 1", transform=plt.gca().transAxes)
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_PDF.pdf", xlabel=cmap_labels[1], ylabel="PDF", ylog=True, xlim=[vmin,vmax])

