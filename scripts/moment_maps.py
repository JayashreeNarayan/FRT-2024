#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from astropy import constants as c

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
    npix = data.shape[0]
    return cfp.gauss_smooth(data, fwhm=npix/2, mode='nearest')

def get_vmin_vmax_centred(data):
    minmax = np.max([-data.min(), data.max()])
    return -minmax, +minmax

def resize_45(data, choice):
    if choice == '2D':
        rescaled_data = np.zeros((128,128))
        for i in range(0,128):
            K = data[:,i]
            rescaled_data[i] = K[25:153]
        return rescaled_data
    elif choice == '3D':
        A = list(range(0,25))
        B = list(range(128,153))
        rescaled_data = np.delete(data, A, 0)
        rescaled_data = np.delete(rescaled_data, B, 0)
        return rescaled_data

def PDF_img_names(i, sigma):
    img_names = ["Optically thin", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
    return img_names[i]+" "+r"; $\sigma$ = "+str(sigma)+r"$~\mathrm{km\,s^{-1}}$"

def func(x,a,n):
    return a*(x**n)

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Plot files')

    choices = ['othin', 'flash', 'ppv_10', 'ppv_21']
    parser.add_argument('-a', '--action', metavar='action', nargs='*', default=choices, choices=choices,
                        help='Choice: Between plotting the first moment maps with flashplotlib directly from the FLASH data (flash), plotting the optically thin first moment maps with cfpack (othin) and plotting the optically thick moment maps from PPV cubes (ppv)')

    args = parser.parse_args()

    # set paths and create plot directory
    path = "../Data/"
    outpath = "../plots/"
    if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

    # set some global option/variables
    xmin=-0.45
    xmax=+0.45
    ymin=1.e-2
    ymax=1.e2

    # Fourier labels
    FT_xy_labels = [r"$k$", r"$P_\mathrm{tot}$"]
    angles = [0, 45, 90]
    FTdata = []
    FTdata_raw = []
    a=10**(-40); n=2.5*10**(-5)
    FT_dict={a:10**(-40), n:2.5*10**(-5)}
    FTdata_SE = []
    
    # LOS labels positions
    LOS_labels_xpos = 0.75
    LOS_labels_ypos = 0.82
    LOS_PDF_labels_xpos = 0.82
    LOS_PDF_labels_ypos = 0.75

    # image title positions
    img_names_xpos = 0.05
    img_names_ypos = 0.9
    img_PDF_names_xpos = 0.02
    img_PDF_names_ypos = 0.85

    # defining the min and max of the maps universally so that all of them can be compared
    vmin_0 = 0. # zeroth moment map
    vmax_0 = 8.0
    vmin_1 = -0.45 # 1st
    vmax_1 = 0.45
    vmin_2 = 0. # 2nd
    vmax_2 = 0.7

    # defining kmin and kmax for FT spectra graphs:
    kmin = 3
    kmax = 40

    # PDF objects and sigmas
    PDF_obj_bins = []
    PDF_obj_pdf = []
    sigma = []
    PDF_obj_bins_corrected = []
    PDF_obj_pdf_corrected = []
    sigma_corrected = []
    linestyle=['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted']

    moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
    cmaps = ['plasma', 'seismic', 'viridis']
    cmap_labels = [r"${I/\langle I \rangle}$", r"${{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", r"${\sigma_{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$"]
    LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
    xyzlabels = [r"$x$", r"$y$", r"$z$", r"$\sqrt{x^2 + z^2}$"]
    img_names = ["Optically thin", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]

    # loop through chosen actions
    for action in args.action:
        print("=== Working on action '"+action+"' ===", color='green')

        # Plotting the optically thin first moment maps with cfpack and also smoothing them out and then obtaining the Gaussian-corrected maps
        if action == choices[0]: 
            files = ["FMM_00.0_0.0.npy", "FMM_45.0_0.0.npy", "FMM_90.0_0.0.npy", "SMM_00.0_0.0.npy", "SMM_45.0_0.0.npy", "SMM_90.0_0.0.npy", "ZMM_00.0_0.0.npy", "ZMM_45.0_0.0.npy", "ZMM_90.0_0.0.npy"]
            for file in files:
                data = np.load(path+"/Data_1tff/Othin/"+file)
                
                if get_LOS(file) == 0: # theta is 0 degrees - along Z axis
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[0] # x-axis on the vertical
                
                if get_LOS(file) == 2: # theta is 90 degrees - along X axis 
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[2] # z-axis on the vertical 

                # for the files with 45 degrees -  we have to resize the data
                if get_LOS(file) == 1: # this means that theta is 45 degrees
                    data = resize_45(data, "2D")
                    xlabel = xyzlabels[1] # y-axis on the bottom 
                    ylabel = xyzlabels[3] # combination of x and z on the vertical

                    if file[:1] == "S": # Since the 2nd moment map needs different plot variables
                        cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2, colorbar = False, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) # for img_name
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                        # plotting the same with individual colorbars for Fig. 1 eqv.
                        cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2, colorbar = True, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                    elif file[:1] == "Z": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                        data = rescale_data(data)
                        cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0, colorbar = False, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                        # plotting the same with individual colorbars for Fig.1 eqv
                        cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0, colorbar = True, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                    else: # all the first moment maps - for Appen. fig
                        cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                        # Also plotting the same with colorbars for Fig. 1
                        cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                        # Creating the PDF for the optically thin case - 1st-moment, without filtering; only for the 45 degrees case
                        if get_LOS(file) == 1:
                            K = cfp.get_pdf(data)
                            PDF_obj_bins.append(K.bin_edges)
                            PDF_obj_pdf.append(K.pdf)
                            sigma.append(round(np.std(data),3))
                    
                # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
                if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or gaussian corrected 
                    smooth_data = smoothing(data)

                    # Gaussian-correction of the smoothed data
                    corrected_data_othin = data - smooth_data
                    if get_LOS(file) == 1:
                        cfp.plot_map(corrected_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes) 
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                    # For FMM summary plot
                    if get_LOS(file) == 0 : axis = ["",None]
                    if get_LOS(file) == 1 : axis = ["",""]
                    if get_LOS(file) == 2 : axis = ["",""]
                    cfp.plot_map(corrected_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    t = plt.text(LOS_labels_xpos, LOS_labels_ypos, LOS_labels[get_LOS(file)] , transform=plt.gca().transAxes) # for LOS
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    if axis[0] == "": xlabel = ""
                    if axis[0] == None: xlabel = xlabel
                    if axis[1] == "": ylabel = ""
                    if axis[1] == None: ylabel = ylabel
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_FMM_sum.pdf")

                    # Fourier Analysis Data for Optically thin maps
                    if get_LOS(file) == 1:
                        K_P = fourier_spectrum(corrected_data_othin)
                        FTdata.append(K_P)
                        K_P_raw = fourier_spectrum(data)
                        FTdata_raw.append(K_P_raw)

                        # for the PDFs
                        K = cfp.get_pdf(corrected_data_othin)
                        PDF_obj_bins_corrected.append(K.bin_edges)
                        PDF_obj_pdf_corrected.append(K.pdf)
                        sigma_corrected.append(round(np.std(corrected_data_othin),3))

            # Generating graphs for For SimEnd time 
            files = ["FMM_45.npy", "SMM_45.npy", "ZMM_45.npy"]
            for file in files:
                data = np.load(path+"/Data_SimEnd/Othin/"+file)
                data = resize_45(data, "2D")
                xlabel = xyzlabels[1] # y-axis on the bottom 
                ylabel = xyzlabels[3] # combination of x and z on the vertical

                if file[:1] == "S": # Since the 2nd moment map needs different plot variables
                    cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2, colorbar = False, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) # for img_name
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"SE.pdf")

                    # plotting the same with individual colorbars for Fig. 1 eqv.
                    cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2, colorbar = True, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb_SE.pdf") # cb = 'colorbar'

                elif file[:1] == "Z": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                    data = rescale_data(data)
                    cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0, colorbar = False, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"SE.pdf")

                    # plotting the same with individual colorbars for Fig.1 eqv
                    cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0, colorbar = True, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb_SE.pdf") # cb = 'colorbar'

                else: # all the first moment maps - for Appen. fig
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-4]+"SE.pdf")

                    # Also plotting the same with colorbars for Fig. 1
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb_SE.pdf") # cb = 'colorbar'

                    # Creating the PDF for the optically thin case - 1st-moment, without filtering; only for the 45 degrees case
                    if get_LOS(file) == 1:
                        K = cfp.get_pdf(data)
                        PDF_obj_bins.append(K.bin_edges)
                        PDF_obj_pdf.append(K.pdf)
                        sigma.append(round(np.std(data),3))
                
            # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
            if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or gaussian corrected 
                smooth_data = smoothing(data)

                # Gaussian-correction of the smoothed data
                corrected_data_othin = data - smooth_data
                cfp.plot_map(corrected_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes) 
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_SE.pdf")

                # Fourier Analysis Data for Optically thin maps
                K_P = fourier_spectrum(corrected_data_othin)
                FTdata_SE.append(K_P)

        # Plotting the zeroth moment maps with flashplotlib directly from the FLASH data , used only in Fig. 1 so we need a colorbar
        '''
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -d vel"+dir+" -nolog -cmap seismic -mw -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"velocity\" -time_scale 0"
                #cmd = "flashplotlib.py -i "+file+" -nolog -cmap plasma -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"Density (g/cm$^3$)\" -time_scale 0"
                cfp.run_shell_command(cmd)
        '''
        # PPV cubes for CO (1-0) lines - 0 moment map and consequently first moment map; smoothing and also gaussian correction
        if action == choices[2]:

            files = ["PPV_00.0_0.npy", "PPV_45.0_0.npy", "PPV_90.0_0.npy"] 
            for file in files:
                moms = [] # empty list to store all the moment maps

                # read PPV data and V axis
                PPV = np.load(path+"/Data_1tff/"+file) # loading the data
                Vrange = np.load(path+"/Data_1tff/"+"Vrange.npy")
                
                if get_LOS(file) == 0: # theta is 0 degrees - along Z axis
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[0] # x-axis on the vertical

                # for the files with 45 degrees -  we have to resize the data
                if get_LOS(file) == 1: # this means that theta is 45 degrees
                    PPV = resize_45(PPV, "3D")
                    xlabel = xyzlabels[1] # y-axis on the bottom 
                    ylabel = xyzlabels[3] # combination of x and z on the vertical

                if get_LOS(file) == 2: # theta is 90 degrees - along X axis 
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[2] # z-axis on the vertical

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange); mom = rescale_data(mom)  # need to rescale the 0th moment map alone
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the optically thin images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)        
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        if get_LOS(file) == 0 | get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2
                        if get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for only the 45.0
                    
                    if get_LOS(file) == 1:
                        # For Appen. Fig.
                        cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))                      
                        cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_"+moment_map+".pdf")

                        # Set with individual colorbars (Fig. 1)
                        cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                        # plotting a common colorbar, only for seismic, universal vmin and vmax
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar.pdf", panels=2)
            
                # Smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1

                # Generating corrected map and then plotting it
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                if get_LOS(file) == 1:
                    cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                # For Fig. 4 - FMM summary plot
                if get_LOS(file) == 0 : axis = ["",None]
                if get_LOS(file) == 1 : axis = ["",""]
                if get_LOS(file) == 2 : axis = ["",""]
                cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                t = plt.text(LOS_labels_xpos, LOS_labels_ypos, LOS_labels[get_LOS(file)] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if axis[0] == "": xlabel = ""
                if axis[0] == None: xlabel = xlabel
                if axis[1] == "": ylabel = ""
                if axis[1] == None: ylabel = ylabel
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_FMM_sum.pdf")

                # Fourier Analysis Data for Synthetic CO
                if get_LOS(file) == 1:
                    K_P = fourier_spectrum(corrected_data)
                    FTdata.append(K_P)
                    K_P_raw = fourier_spectrum(moms[1])
                    FTdata_raw.append(K_P_raw)
                        
                    # Make PDF of orginal mom1 and plot
                    K = cfp.get_pdf(moms[1])
                    PDF_obj_bins.append(K.bin_edges)
                    PDF_obj_pdf.append(K.pdf)
                    sigma.append(round(np.std(moms[1]),3))

                    # Make PDF of low-pass-filtered moment 1 and also plot it;  only for the 45 degrees case
                    K = cfp.get_pdf(corrected_data)
                    PDF_obj_bins_corrected.append(K.bin_edges)
                    PDF_obj_pdf_corrected.append(K.pdf)
                    sigma_corrected.append(round(np.std(corrected_data),3))
            
            # Doing the same as above for Data_SimEnd
            files = ["PPV_45.npy"] 
            for file in files:
                moms = [] # empty list to store all the moment maps

                # read PPV data and V axis
                PPV = np.load(path+"/Data_SimEnd/"+file) # loading the data
                Vrange = np.load(path+"/Data_SimEnd/"+"Vrange.npy")           
                PPV = resize_45(PPV, "3D")
                xlabel = xyzlabels[1] # y-axis on the bottom 
                ylabel = xyzlabels[3] # combination of x and z on the vertical

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange); mom = rescale_data(mom)  # need to rescale the 0th moment map alone
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the optically thin images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)        
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2
                        moms[imom] = moms[imom].T # transpose for only the 45.0
                
                    # For Appen. Fig.
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))                      
                    cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_"+moment_map+"_SE.pdf")

                    # Set with individual colorbars (Fig. 1)
                    cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_map+"_cb_SE.pdf")
        
                # Smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1

                # Generating corrected map and then plotting it
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected_SE.pdf")

                # Fourier Analysis Data for Synthetic CO
                K_P = fourier_spectrum(corrected_data)
                FTdata_SE.append(K_P)
            
        # PPV cubes for CO (2-1) lines - 0 moment map and consequently first moment map; smoothing and also gaussian correction
        if action == choices[3]:

            files = ["PPV_00.0_0_J21.npy", "PPV_45.0_0_J21.npy", "PPV_90.0_0_J21.npy"] 
            for file in files:
                moms = [] # empty list to store all the moment maps

                # read PPV data and V axis
                PPV = np.load(path+"Data_1tff/J21/"+file) # loading the data
                Vrange = np.load(path+"/Data_1tff/"+"Vrange.npy")
                
                if get_LOS(file) == 0: # theta is 0 degrees - along Z axis
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[0] # x-axis on the vertical

                # for the files with 45 degrees -  we have to resize the data
                if get_LOS(file) == 1: # this means that theta is 45 degrees
                    PPV = resize_45(PPV, "3D")
                    xlabel = xyzlabels[1] # y-axis on the bottom 
                    ylabel = xyzlabels[3] # combination of x and z on the vertical

                if get_LOS(file) == 2: # theta is 90 degrees - along X axis 
                    xlabel = xyzlabels[1] # y-axis on the horizontal
                    ylabel = xyzlabels[2] # z-axis on the vertical

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange); mom = rescale_data(mom)  # need to rescale the 0th moment map alone
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the optically thin images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)        
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        if get_LOS(file) == 0 | get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2
                        if get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for only the 45.0
                    
                    if get_LOS(file) == 1:
                        # For Appen. Fig.
                        cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))                     
                        cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_"+moment_map+".pdf")

                        # Set with individual colorbars (Fig. 1)
                        cfp.plot_map(moms[imom], cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                        t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                        # plotting a common colorbar, only for seismic, universal vmin and vmax
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar.pdf", panels=2)
        
                # Smoothing (low-pass filtering) of moment 1
                print("Now doing low-pass filter on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1

                # Generating corrected map and then plotting it
                print("Now subtracting low-pass-filtered moment 1")
                corrected_data = moms[1] - smooth_mom1 # subtraction
                if get_LOS(file) == 1:
                    cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_corrected.pdf")

                # For Fig. 4 - FMM summary plot
                if get_LOS(file) == 0 : axis = [None,None]
                if get_LOS(file) == 1 : axis = [None,""]
                if get_LOS(file) == 2 : axis = [None,""]
                cfp.plot_map(corrected_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                t = plt.text(LOS_labels_xpos, LOS_labels_ypos, LOS_labels[get_LOS(file)] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if axis[0] == "": xlabel = ""
                if axis[0] == None: xlabel = xlabel
                if axis[1] == "": ylabel = ""
                if axis[1] == None: ylabel = ylabel
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_FMM_sum.pdf")

                # Fourier Analysis Data for Synthetic CO
                if get_LOS(file) == 1:
                    K_P = fourier_spectrum(corrected_data)
                    FTdata.append(K_P)
                    K_P_raw = fourier_spectrum(moms[1])
                    FTdata_raw.append(K_P_raw)
                        
                    # Make PDF of orginal mom1 and plot
                    K = cfp.get_pdf(moms[1])
                    PDF_obj_bins.append(K.bin_edges)
                    PDF_obj_pdf.append(K.pdf)
                    sigma.append(round(np.std(moms[1]),3))

                    # Make PDF of low-pass-filtered moment 1 and also plot it
                    K = cfp.get_pdf(corrected_data)
                    PDF_obj_bins_corrected.append(K.bin_edges)
                    PDF_obj_pdf_corrected.append(K.pdf)
                    sigma_corrected.append(round(np.std(corrected_data),3))

    # Plotting the FTs - after correction, 1tff
    FTdata = np.array(FTdata, dtype=object)
    for i in range(len(FTdata)):
        cfp.plot(x=FTdata[i,0], y=FTdata[i,1], label=img_names[i], linestyle=linestyle[i])
        #cfp.fit(func, xdat=FTdata[i,0][3:], ydat=FTdata[i,1][3:], params=FT_dict)
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(which = 'major', top=False)
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.5, text=r"1st-moment", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, xlim=[kmin,kmax], save=outpath+"FT_after.pdf")

    # Plotting the FTs - before correction, SimEnd
    FTdata_raw = np.array(FTdata_raw, dtype=object)
    for i in range(len(FTdata_raw)):
        cfp.plot(x=FTdata_raw[i,0], y=FTdata_raw[i,1], label=img_names[i], linestyle=linestyle[i])
        #cfp.fit(func, xdat=FTdata_raw[i,0][3:], ydat=FTdata_raw[i,1][3:], params=FT_dict)
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(which = 'major', top=False)
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.5, text=r"Low-pass filtered", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1],  fontsize='small', ylog=True,  xlog=True, xlim=[kmin,kmax], save=outpath+"FT_before.pdf")
    
    # Plotting the FTs - after correction, 1tff
    FTdata = np.array(FTdata_SE, dtype=object)
    for i in range(len(FTdata_SE)):
        cfp.plot(x=FTdata_SE[i,0], y=FTdata_SE[i,1], label=img_names[i], linestyle=linestyle[i])
        #cfp.fit(func, xdat=FTdata[i,0][3:], ydat=FTdata[i,1][3:], params=FT_dict)
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(which = 'major', top=False)
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.5, text=r"1st-moment", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, xlim=[kmin,kmax], save=outpath+"FT_after_SE.pdf")

    # Plotting the PDFs
    # for optically thin case
    cfp.plot(x=PDF_obj_bins[0], y=PDF_obj_pdf[0], type="pdf", bar_width=1, label=PDF_img_names(0, sigma[0]), linestyle=linestyle[0])
    cfp.plot(x=PDF_obj_bins[1], y=PDF_obj_pdf[1], type="pdf", bar_width=1, label=PDF_img_names(1, sigma[1]), linestyle=linestyle[1])
    cfp.plot(x=PDF_obj_bins[2], y=PDF_obj_pdf[2], type="pdf", bar_width=1, label=PDF_img_names(2, sigma[2]), linestyle=linestyle[2])
    
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=r"1st-moment", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"before_correction_PDF.pdf")

    # for optically Thick case
    cfp.plot(x=PDF_obj_bins_corrected[0], y=PDF_obj_pdf_corrected[0], type="pdf", bar_width=1, label=PDF_img_names(0, sigma_corrected[0]), linestyle=linestyle[0])
    cfp.plot(x=PDF_obj_bins_corrected[1], y=PDF_obj_pdf_corrected[1], type="pdf", bar_width=1, label=PDF_img_names(1, sigma_corrected[1]), linestyle=linestyle[1])
    cfp.plot(x=PDF_obj_bins_corrected[2], y=PDF_obj_pdf_corrected[2], type="pdf", bar_width=1, label=PDF_img_names(2, sigma_corrected[2]), linestyle=linestyle[2])
    
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=r"Low-pass filtered", backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"after_correction_PDF.pdf")
