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
    sigma=cfp.round(sigma, 2, str_ret=True)
    img_names = ["Optically thin", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
    return img_names[i]+r": $\sigma$ = "+sigma+r"$~\mathrm{km\,s^{-1}}$"

def FT_slope_labels(err,n):
    err_n="0.1"
    return ";~slope="+n+r"$\pm$"+err_n

def func(x,a,n):
    return np.log(a*(x**n))

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

    # For the PDFs
    xmin=-0.45
    xmax=+0.45
    ymin=1.e-2
    ymax=5.e2

    # Fourier labels
    FT_xy_labels = [r"$k$", r"$P_\mathrm{tot}$"]
    angles = [0, 45, 90]
    FTdata = []
    FTdata_raw = []
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
    vmax_0 = 4
    vmin_1 = -0.55 # 1st
    vmax_1 = 0.55
    vmin_2 = 0. # 2nd
    vmax_2 = 0.6

    # defining the min and max of the maps - for SimEnd alone
    vmax_0_SE = 6
    vmax_2_SE = 0.6

    # defining kmin and kmax for FT spectra graphs:
    kmin = 3
    kmax = 40

    # PDF objects and sigmas
    PDF_obj_bins = [] # for before isolation
    PDF_obj_pdf = []
    sigma = []

    PDF_obj_bins_isolated = [] # for after isolation
    PDF_obj_pdf_isolated = []
    sigma_isolated = []

    PDF_obj_bins_SE = [] # for SE case
    PDF_obj_pdf_SE = []
    sigma_SE = []
    
    all_sigmas =[]
    all_sigmas_before = []

    linestyle=['dotted', 'dashed', 'dashdot', 'loosely dotted']
    line_colours=['black', 'magenta', 'blue']

    moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
    cmaps = ['plasma', 'seismic', 'viridis']
    cmap_labels = [r"${I/\langle I \rangle}$", r"${{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", r"${\sigma_{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$"]
    LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
    xyzlabels = [r"$x$", r"$y$", r"$z$", r"$\sqrt{x^2 + z^2}$"]
    img_names = ["Optically thin", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
    img_types=['Before turbulence isolation', 'After turbulence isolation']

    # loop through chosen actions
    for action in args.action:
        print("=== Working on action '"+action+"' ===", color='green')

        # Plotting the optically thin first moment maps with cfpack and also smoothing them out and then obtaining the Turbulence-isolated maps
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
                    
                # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
                if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
                    smooth_data = smoothing(data)
                    all_sigmas_before.append((file , np.std(data)))
                    # Turbulence isolation of the smoothed data
                    isolated_data_othin = data - smooth_data
                    all_sigmas.append((file , np.std(isolated_data_othin)))
                    if get_LOS(file) == 1:
                        cfp.plot_map(isolated_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) 
                        t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                        cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_isolated.pdf")

                    # For FMM summary plot
                    if get_LOS(file) == 0 : axis = ["",None]
                    if get_LOS(file) == 1 : axis = ["",None]
                    if get_LOS(file) == 2 : axis = [None,None]
                    cfp.plot_map(isolated_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
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
                        K_P = fourier_spectrum(isolated_data_othin) # for the turbulence isolated data
                        FTdata.append(K_P)
                        K_P_raw = fourier_spectrum(data) # for the unisolated data
                        FTdata_raw.append(K_P_raw)

                        # for the PDFs
                        K = cfp.get_pdf(data, range=(-0.1, 0.1)) # without isolation
                        PDF_obj_bins.append(K.bin_edges)
                        PDF_obj_pdf.append(K.pdf)
                        sigma.append(np.std(data))
                        K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1)) # with isolation
                        PDF_obj_bins_isolated.append(K.bin_edges)
                        PDF_obj_pdf_isolated.append(K.pdf)
                        sigma_isolated.append(np.std(isolated_data_othin))

            # Generating graphs for For SimEnd time 
            files = ["FMM_45.0_SE.npy", "SMM_45.0_SE.npy", "ZMM_45.0_SE.npy"]
            for file in files:
                data = np.load(path+"/Data_SimEnd/Othin/"+file)
                data = resize_45(data, "2D")
                xlabel = xyzlabels[1] # y-axis on the bottom 
                ylabel = xyzlabels[3] # combination of x and z on the vertical

                if file[:1] == "S": # Since the 2nd moment map needs different plot variables
                    cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2_SE, colorbar = False, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) # for img_name
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+".pdf")

                    # plotting the same with individual colorbars for Fig. 1 eqv.
                    cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2_SE, colorbar = True, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                elif file[:1] == "Z": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                    data = rescale_data(data)
                    cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0_SE, colorbar = False, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+".pdf")

                    # plotting the same with individual colorbars for Fig.1 eqv
                    cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0_SE, colorbar = True, cmap_label=cmap_labels[0], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                else: # all the first moment maps - for Appen. fig
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-4]+".pdf")

                    # Also plotting the same with colorbars for Fig. 1
                    cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'
                    
                # Smoothing of the optically thin moment maps - done only for moment 1 maps, skipping moment 2 data
                if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
                    smooth_data = smoothing(data)
                    all_sigmas_before.append(('SE', file , np.std(data)))
                    # Turbulence isolation of the smoothed data
                    isolated_data_othin = data - smooth_data
                    all_sigmas.append(('SE', file , np.std(isolated_data_othin)))
                    cfp.plot_map(isolated_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) 
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_isolated.pdf")

                    # Fourier Analysis Data for Optically thin maps
                    K_P = fourier_spectrum(isolated_data_othin)
                    FTdata_SE.append(K_P)
                    
                    # PDFs for after corection alone - SimEnd Case
                    if get_LOS(file) == 1:
                        K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1))
                        PDF_obj_bins_SE.append(K.bin_edges)
                        PDF_obj_pdf_SE.append(K.pdf)
                        sigma_SE.append(np.std(isolated_data_othin))

        # Plotting the zeroth moment maps with flashplotlib directly from the FLASH data , used only in Fig. 1 so we need a colorbar
        '''
        if action == choices[1]:
            file = path + "ChemoMHD_hdf5_plt_cnt_0001"
            for dir in ['x', 'y', 'z']:
                cmd = "flashplotlib.py -i "+file+" -d vel"+dir+" -nolog -cmap seismic -mw -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"velocity\" -time_scale 0"
                #cmd = "flashplotlib.py -i "+file+" -nolog -cmap plasma -direction "+dir+" -outtype pdf -outdir "+outpath+" -cmap_label \"Density (g/cm$^3$)\" -time_scale 0"
                cfp.run_shell_command(cmd)
        '''
        # PPV cubes for CO (1-0) lines - 0 moment map and consequently first moment map; smoothing and also turbulence isolation
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
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p1.pdf", panels=1) 
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p2.pdf", panels=2)
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p3.pdf", panels=3)
            
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append((file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction
                all_sigmas.append((file, np.std(isolated_data)))

                if get_LOS(file) == 1:
                    cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

                # For A1 - FMM summary plot
                if get_LOS(file) == 0 : axis = ["",""]
                if get_LOS(file) == 1 : axis = ["",""]
                if get_LOS(file) == 2 : axis = [None,""]
                cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                t = plt.text(LOS_labels_xpos, LOS_labels_ypos, LOS_labels[get_LOS(file)] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if axis[0] == "": xlabel = ""
                if axis[0] == None: xlabel = xlabel
                if axis[1] == "": ylabel = ""
                if axis[1] == None: ylabel = ylabel
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_FMM_sum.pdf")

                # Fourier Analysis Data for Synthetic CO (1-0)
                if get_LOS(file) == 1:
                    K_P = fourier_spectrum(isolated_data)
                    FTdata.append(K_P)
                    K_P_raw = fourier_spectrum(moms[1])
                    FTdata_raw.append(K_P_raw)

                    # Make PDF of turbulence isolated moment 1 and also plot it;  only for the 45 degrees case
                    K = cfp.get_pdf(moms[1], range=(-0.1,+0.1)) # for before turbulence isolation
                    PDF_obj_bins.append(K.bin_edges)
                    PDF_obj_pdf.append(K.pdf)
                    sigma.append(np.std(moms[1]))

                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # for after turbulence isolation
                    PDF_obj_bins_isolated.append(K.bin_edges)
                    PDF_obj_pdf_isolated.append(K.pdf)
                    sigma_isolated.append(np.std(isolated_data))
            
            # Doing the same as above for Data_SimEnd
            files = ["PPV_45.0.npy"] 
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
                        vmax = vmax_0_SE
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2_SE
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
        
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append(('SE', file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction
                all_sigmas.append(('SE', file, np.std(isolated_data)))

                cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated_SE.pdf")

                # Fourier Analysis Data for Synthetic CO - SimEnd
                K_P = fourier_spectrum(isolated_data)
                FTdata_SE.append(K_P)
                
                # Getting the PDF for SimEnd - CO (1-0)
                if get_LOS(file) == 1:
                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1))
                    PDF_obj_bins_SE.append(K.bin_edges)
                    PDF_obj_pdf_SE.append(K.pdf)
                    sigma_SE.append(np.std(isolated_data))
            
        # PPV cubes for CO (2-1) lines - 0 moment map and consequently first moment map; smoothing and also turbulence isolation
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
        
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append((file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction
                all_sigmas.append((file, np.std(isolated_data)))
                if get_LOS(file) == 1:
                    cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                    t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                    cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

                # For Appendix Fig. - FMM summary plot
                if get_LOS(file) == 0 : axis = ["",""]
                if get_LOS(file) == 1 : axis = ["",""]
                if get_LOS(file) == 2 : axis = [None,""]
                cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=axis, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                t = plt.text(LOS_labels_xpos, LOS_labels_ypos, LOS_labels[get_LOS(file)] , transform=plt.gca().transAxes) # for LOS
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                if axis[0] == "": xlabel = ""
                if axis[0] == None: xlabel = xlabel
                if axis[1] == "": ylabel = ""
                if axis[1] == None: ylabel = ylabel
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_FMM_sum.pdf")

                # Fourier Analysis Data for Synthetic CO (2-1)
                if get_LOS(file) == 1:
                    K_P = fourier_spectrum(isolated_data)
                    FTdata.append(K_P)
                    K_P_raw = fourier_spectrum(moms[1])
                    FTdata_raw.append(K_P_raw)
                        
                    # Make PDF of turbulence isolated moment 1 and also plot it
                    K = cfp.get_pdf(moms[1], range=(-0.1,+0.1)) # before turbulence isolation
                    PDF_obj_bins.append(K.bin_edges)
                    PDF_obj_pdf.append(K.pdf)
                    sigma.append(np.std(moms[1]))

                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # after turbulence isolation
                    PDF_obj_bins_isolated.append(K.bin_edges)
                    PDF_obj_pdf_isolated.append(K.pdf)
                    sigma_isolated.append(np.std(isolated_data))
            
            # Doing the same as above for Data_SimEnd
            files = ["PPV_45.0_J21_SE.npy"] 
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
                        vmax = vmax_0_SE
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2_SE
                        moms[imom] = moms[imom].T # transpose for only the 45.0
                
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
        
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append(('SE', file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction
                all_sigmas.append(('SE', file, np.std(isolated_data)))

                cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[2] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

                # Fourier Analysis Data for Synthetic CO (2-1) - SimEnd
                K_P = fourier_spectrum(isolated_data)
                FTdata_SE.append(K_P)
                
                # Getting the PDF for SimEnd - CO (2-1) - SimEnd
                K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # only after turbulence isolation
                PDF_obj_bins_SE.append(K.bin_edges)
                PDF_obj_pdf_SE.append(K.pdf)
                sigma_SE.append(np.std(isolated_data))

    # Plotting the FTs - before isolation, 1tff
    for i in range(len(FTdata_raw)):
        x=FTdata_raw[i][0][1:kmax]; y=FTdata_raw[i][1][1:kmax]
        params = {"a":[1e-4, 1e-2, 1], "n":[-4, -2, -1]}
        fit_values = cfp.fit(func, xdat=x[kmin:], ydat=np.log(y[kmin:]), perr_method='systematic', params=params)
        a=cfp.round(fit_values.popt[0], 2, str_ret=True); n=cfp.round(fit_values.popt[1], 2, str_ret=True)
        cfp.plot(x=x, y=y, label=img_names[i]+FT_slope_labels(fit_values.perr,n), color=line_colours[i])
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle=linestyle[0])
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(axis='x', direction='in', length=0, which = 'minor', top=False, bottom=True)
    cfp.plot(x=img_PDF_names_xpos+0.003, y=img_PDF_names_ypos-0.55, text=img_types[0], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, save=outpath+"FT_before.pdf")
    
    # Plotting the FTs - after isolation, 1tff
    for i in range(len(FTdata)):
        x=FTdata[i][0][1:kmax]; y=FTdata[i][1][1:kmax]
        params = {"a":[1e-4, 1e-2, 1], "n":[-4, -2, -1]}
        fit_values = cfp.fit(func, xdat=x[kmin:], ydat=np.log(y[kmin:]), perr_method='systematic', params=params)
        a=cfp.round(fit_values.popt[0], 2, str_ret=True); n=cfp.round(fit_values.popt[1], 2, str_ret=True)
        cfp.plot(x=x, y=y, label=img_names[i]+FT_slope_labels(fit_values.perr,n), color=line_colours[i])
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle=linestyle[0])
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(axis='x', direction='in', length=0, which = 'minor', top=False, bottom=True)
    cfp.plot(x=img_PDF_names_xpos+0.003, y=img_PDF_names_ypos-0.55, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, save=outpath+"FT_after.pdf")
    
    # Plotting the FTs - after isolation, SimEnd
    for i in range(len(FTdata_SE)):
        x=FTdata_SE[i][0][1:kmax]; y=FTdata_SE[i][1][1:kmax]
        params = {"a":[1e-4, 1e-2, 1], "n":[-4, -2, -1]}
        fit_values = cfp.fit(func, xdat=x[kmin:], ydat=np.log(y[kmin:]), perr_method='systematic', params=params)
        a=cfp.round(fit_values.popt[0], 2, str_ret=True); n=cfp.round(fit_values.popt[1], 2, str_ret=True)
        cfp.plot(x=x, y=y, label=img_names[i]+FT_slope_labels(fit_values.perr,n), color=line_colours[i])
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle=linestyle[0])
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(axis='x', direction='in', length=0, which = 'minor', top=False, bottom=True)
    cfp.plot(x=img_PDF_names_xpos+0.003, y=img_PDF_names_ypos-0.55, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, save=outpath+"FT_after_SE.pdf")

    # Plotting the PDFs for before isolation
    for i in range(len(PDF_obj_bins)):
        cfp.plot(x=PDF_obj_bins[i], y=PDF_obj_pdf[i], type='pdf', label=PDF_img_names(i, sigma[i]), color=line_colours[i])
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=img_types[0], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"before_isolation_PDF.pdf")

    # Plotting the PDFs for after isolation
    for i in range(len(PDF_obj_bins_isolated)):
        cfp.plot(x=PDF_obj_bins_isolated[i], y=PDF_obj_pdf_isolated[i], type='pdf', label=PDF_img_names(i, sigma_isolated[i]), color=line_colours[i])
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"after_isolation_PDF.pdf")

    # Plotting the PDFs for after isolation - FOR SIMEND
    for i in range(len(PDF_obj_bins_SE)):    
        cfp.plot(x=PDF_obj_bins_SE[i], y=PDF_obj_pdf_isolated[i], type='pdf', label=PDF_img_names(i, sigma_SE[i]), color=line_colours[i])
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"after_isolation_SE_PDF.pdf")

    # Printing sigma values for Table 1.
    SE=[]; deg_0=[]; deg_45=[]; deg_90=[]; rel_valb=[]; rel_vala=[]
    for i in range(0,4):    
        rel_vala.append(all_sigmas[i][-1])
        rel_valb.append(all_sigmas_before[i][-1])
    for i in range(len(all_sigmas)):
        if all_sigmas[i][0]=='SE':
            sigma_before=cfp.round(all_sigmas_before[i][-1], 2, str_ret=True)
            sigma_after=cfp.round(all_sigmas[i][-1], 2, str_ret=True)
            f_rel_before=cfp.round(all_sigmas_before[i][-1]/rel_valb[3], 2, str_ret=True)
            f_rel_after=cfp.round(all_sigmas[i][-1]/rel_vala[3], 2, str_ret=True)
            SE.append(sigma_before+" & "+f_rel_before+" & "+sigma_after+" & "+f_rel_after)
        else:
            LOS=get_LOS(all_sigmas_before[i][0])
            if LOS==0: rel_vb=rel_valb[LOS]; rel_va=rel_vala[LOS]
            if LOS==1: rel_vb=rel_valb[LOS]; rel_va=rel_vala[LOS]
            if LOS==2: rel_vb=rel_valb[LOS]; rel_va=rel_vala[LOS]
            sigma_before=cfp.round(all_sigmas_before[i][-1], 2, str_ret=True)
            sigma_after=cfp.round(all_sigmas[i][-1], 2, str_ret=True)
            f_rel_before=cfp.round(all_sigmas_before[i][-1]/rel_vb, 2, str_ret=True)
            f_rel_after=cfp.round(all_sigmas[i][-1]/rel_va, 2, str_ret=True)
            if LOS==0: deg_0.append(sigma_before+" & "+f_rel_before+" & "+sigma_after+" & "+f_rel_after)
            if LOS==1: deg_45.append(sigma_before+" & "+f_rel_before+" & "+sigma_after+" & "+f_rel_after)
            if LOS==2: deg_90.append(sigma_before+" & "+f_rel_before+" & "+sigma_after+" & "+f_rel_after)

    print(SE)
    print(deg_0)
    print(deg_45)
    print(deg_90)

        

