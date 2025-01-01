#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from util_scripts.all_functions import *

# ===== the following applies in case we are running this in script mode =====
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Plot files')

    choices = ['Ideal', 'flash', 'ppv_10', 'ppv_21']
    parser.add_argument('-a', '--action', metavar='action', nargs='*', default=choices, choices=choices,
                        help='Choice: Between plotting the first moment maps with flashplotlib directly from the FLASH data (flash), plotting the Ideal first moment maps with cfpack (Ideal) and plotting the optically thick moment maps from PPV cubes (ppv)')

    args = parser.parse_args()

    # set paths and create plot directory
    path = "../Data/"
    outpath = "../plots_old/"
    if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

    # For the correction factor maps, all BEFORE isolation
    ideal_1tff_before = []
    ideal_SE_before = []
    CO_10_1tff_before = []
    CO_10_SE_before = []
    CO_21_1tff_before = []
    CO_21_SE_before = []
    PDF_correction_obj_1_before = []
    correction_sigmas_1_before = []

    # For the correction factor maps, all after isolation
    ideal_1tff = []
    ideal_SE = []
    CO_10_1tff = []
    CO_10_SE = []
    CO_21_1tff = []
    CO_21_SE = []
    PDF_correction_obj_1_after = []
    correction_sigmas_1_after = []

    #   MOMENT 0 CORRECTION #
    ideal_1tff_mom0 = []
    ideal_SE_mom0 = []
    CO_10_1tff_mom0 = []
    CO_10_SE_mom0 = []
    CO_21_1tff_mom0 = []
    CO_21_SE_mom0 = []
    PDF_correction_obj_0 = []
    correction_sigmas_0 = []

    #   MOMENT 2 CORRECTION #
    ideal_SE_mom2 = []
    CO_10_SE_mom2 = []
    CO_10_1tff_mom2=[]
    CO_21_1tff_mom2 = []
    CO_21_SE_mom2 = []
    PDF_correction_obj_2 = []
    correction_sigmas_2 = []

    # vmin and vmax for correction factors
    vmin_correc = -1000
    vmax_correc = 1000
    ylim_min = 1.e-6
    ylim_max = 100

    # For the correction factor PDFs, all after isolation
    correction_labels = [r"$\mathrm{CO}\,(1-0)$ at $1\,t_\mathrm{ff}$", r"$\mathrm{CO}\,(1-0)$ at $1.2\,t_\mathrm{ff}$", r"$\mathrm{CO}\,(2-1)$ at $1\,t_\mathrm{ff}$", r"$\mathrm{CO}\,(2-1)$ at $1.2\,t_\mathrm{ff}$"]
    correction_cmap_lables =[
    r"$\left( \frac{I}{\langle I \rangle} \right)_{\mathrm{CO}} / \left( \frac{I}{\langle I \rangle} \right)_{\mathrm{Ideal}}$", r'$I_{\mathrm{norm,}\,\mathrm{CO}\,(2-1) / I_{\mathrm{norm,}\, \mathrm{Ideal}}}$',
    r"${\sigma_{v_{\mathrm{LOS,\ CO}\,(1-0)}}}/ {\sigma_{v_{\mathrm{LOS,\ Ideal}}}}$", r"${\sigma_{v_{\mathrm{LOS,\ CO}\,(2-1)}}} / {\sigma_{v_{\mathrm{LOS,\ Ideal}}}}$", 
    r"${{v_{\mathrm{LOS,\ CO}\,(1-0)}}} - {{v_{\mathrm{LOS,\ Ideal}}}}$", r"${{v_{\mathrm{LOS,\ CO}\,(2-1)}}} - {{v_{\mathrm{LOS,\ Ideal}}}}$" ]
    correction_xlabel = "Correction factors"

    correction_bins_values = [-1000, -900, -800, -700, -600, -500, -400, -300, -200, -100 ,0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    
    # For the PDFs
    xmin=-0.45
    xmax=+0.45
    ymin=1.e-2
    ymax=5.e2
    xfit = np.linspace(xmin, xmax, 500)

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
    PDF_obj = []
    sigma = []
    skewness = []
    kurtosis = []

    PDF_obj_isolated = []
    sigma_isolated = []
    skewness_isolated = []
    kurtosis_isolated = []

    PDF_obj_SE = []
    sigma_SE = []
    skewness_SE_before = []
    kurtosis_SE_before = []
    skewness_SE_after = []
    kurtosis_SE_after = []

    all_sigmas = []
    all_sigmas_before = []

    linestyle = ['dotted', 'dashed', 'dashdot', 'loosely dotted']
    line_colours = ['black', 'magenta', 'blue']

    moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
    cmaps = ['plasma', 'seismic', 'viridis', 'gray', 'PuOr']
    cmap_labels = [r"${I/\langle I \rangle}$", r"${{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", r"${\sigma_{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", "Correction factor values"]
    LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
    xyzlabels = [r"$x$", r"$y$", r"$z$", r"$\sqrt{x^2 + z^2}$"]
    img_names = ["Ideal", r"Synthetic CO$\,(1-0)$", r"Synthetic CO$\,(2-1)$"]
    img_types=['Before turbulence isolation', 'After turbulence isolation']

    # loop through chosen actions
    for action in args.action:
        print("=== Working on action '"+action+"' ===", color='green')

        # Plotting the Ideal first moment maps with cfpack and also smoothing them out and then obtaining the Turbulence-isolated maps
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
                        ideal_1tff_mom2 = data # for the correction factors

                        # plotting the same with individual colorbars for Fig. 1 eqv.
                        ret = cfp.plot_map(np.asarray(data), cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2, colorbar = True, cmap_label=cmap_labels[2], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                    if file[:1] == "Z": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                        data = rescale_data(data)
                        ideal_1tff_mom0.append(data) # for correction maps

                        # plotting the same with individual colorbars for Fig.1 eqv
                        ret = cfp.plot_map(np.asarray(data), cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0, colorbar = True, cmap_label=cmap_labels[0], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'
                    
                # Smoothing of the Ideal moment maps - done only for moment 1 maps, skipping moment 2 data
                if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
                    ideal_1tff_before.append(data)
                    smooth_data = smoothing(data)
                    all_sigmas_before.append((file , np.std(data)))
                    
                    # Turbulence isolation of the smoothed data
                    isolated_data_othin = data - smooth_data
                    all_sigmas.append((file , np.std(isolated_data_othin)))

                    # for the correction factors map
                    ideal_1tff.append(isolated_data_othin)

                    if get_LOS(file) == 1:
                        #producing the smoothed maps
                        ret = cfp.plot_map(np.asarray(smooth_data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlabel="", ylabel=ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_smooth.pdf")

                        ret = cfp.plot_map(np.asarray(isolated_data_othin), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_isolated.pdf")

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

                    # Fourier Analysis Data for Ideal maps
                    if get_LOS(file) == 1:
                        K_P = fourier_spectrum(isolated_data_othin) # for the turbulence isolated data
                        FTdata.append(K_P)
                        K_P_raw = fourier_spectrum(data) # for the unisolated data
                        FTdata_raw.append(K_P_raw)

                        # for the PDFs
                        K = cfp.get_pdf(data, range=(-0.1, 0.1)) # without isolation
                        PDF_obj.append(K)
                        sigma.append(np.std(data))
                        #skewness.append(kurtosis(data.flatten(), 's'))
                        kurtosis.append(kurtosis_own(np.asarray(data).flatten()))

                        K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1)) # with isolation
                        PDF_obj_isolated.append(K)
                        sigma_isolated.append(np.std(isolated_data_othin))
                        #skewness_isolated.append(kurtosis(isolated_data_othin.flatten(), 's'))
                        kurtosis_isolated.append(kurtosis_own(np.asarray(isolated_data_othin).flatten()))

            # Generating graphs for For SimEnd time 
            files = ["FMM_45.0_SE.npy", "SMM_45.0_SE.npy", "ZMM_45.0_SE.npy"]
            for file in files:
                data = np.load(path+"/Data_SimEnd/Othin/"+file)
                data = resize_45(data, "2D")
                xlabel = xyzlabels[1] # y-axis on the bottom 
                ylabel = xyzlabels[3] # combination of x and z on the vertical

                if file[:1] == "S": # Since the 2nd moment map needs different plot variables

                    ideal_SE_mom2.append(data) # for correction maps

                    # plotting the same with individual colorbars for Fig. 1 eqv.
                    ret = cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2_SE, colorbar = True, cmap_label=cmap_labels[2], xlabel=xlabel, ylabel=ylabel,xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                elif file[:1] == "Z": # zeroth moment map has separate plot variables, and is needed only for Fig. 1
                    data = rescale_data(data)
                    ideal_SE_mom0.append(data) # for correction maps

                    # plotting the same with individual colorbars for Fig.1 eqv
                    ret = cfp.plot_map(data, cmap=cmaps[0], vmin=vmin_0, vmax=vmax_0_SE, colorbar = True, cmap_label=cmap_labels[0], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

                else: # all the first moment maps - for Appen. fig
                    # Also plotting the same with colorbars for Fig. 1
                    ret = cfp.plot_map(np.asarray(data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'
                    
                # Smoothing of the Ideal moment maps - done only for moment 1 maps, skipping moment 2 data
                if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
                    ideal_SE_before.append(data)
                    smooth_data = smoothing(data)
                    all_sigmas_before.append(('SE', file , np.std(data)))
                    # Turbulence isolation of the smoothed data
                    isolated_data_othin = data - smooth_data

                    # for the correction factors map
                    ideal_SE.append(isolated_data_othin)
                    all_sigmas.append(('SE', file , np.std(isolated_data_othin)))

                    ret = cfp.plot_map(np.asarray(isolated_data_othin), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None],xlabel=xlabel, ylabel=ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[0], normalised_coords=True)
                    cfp.plot(save=outpath+file[:-4]+"_isolated.pdf")

                    # Fourier Analysis Data for Ideal maps
                    K_P = fourier_spectrum(isolated_data_othin)
                    FTdata_SE.append(K_P)
                    
                    # PDFs for after corection alone - SimEnd Case
                    if get_LOS(file) == 1:
                        K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1))
                        PDF_obj_SE.append(K)
                        sigma_SE.append(np.std(isolated_data_othin))
                        #skewness_SE_before.append(kurtosis(data.flatten(), 's')) # before isolation for SE
                        kurtosis_SE_before.append(kurtosis_own(np.asarray(data).flatten())) # after isolation for SE
                        #skewness_SE_after.append(kurtosis(isolated_data_othin.flatten(), 's'))
                        kurtosis_SE_after.append(kurtosis_own(np.asarray(isolated_data_othin).flatten()))

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
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Ideal images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)        
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0
                        CO_10_1tff_mom0.append(moms[0])
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        if get_LOS(file) == 0 | get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2
                        if get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for only the 45.0
                        CO_10_1tff_mom2.append(moms[2])
                    
                    if get_LOS(file) == 1:
                        # For Appen. Fig.
                        ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""],xlabel="", ylabel="",  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)                                            
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+".pdf")

                        # Set with individual colorbars (Fig. 1)
                        ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True) 
                        cfp.plot(save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                        # plotting a common colorbar, only for seismic, universal vmin and vmax
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p1.pdf", panels=1) 
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p2.pdf", panels=2)
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p3.pdf", panels=3)
                        cfp.plot_colorbar(cmap=cmaps[3], vmin=vmin_correc, vmax=vmax_correc, symlog=True, label=cmap_labels[3], save=outpath+cmaps[3]+"_colorbar_p2.pdf", panels=2) # grey color panel
            
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                CO_10_1tff_before.append(moms[1])
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append((file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction

                # for the correction factors map
                CO_10_1tff.append(isolated_data)
                
                all_sigmas.append((file, np.std(isolated_data)))

                if get_LOS(file) == 1:
                    # getting the smoothed maps
                    ret = cfp.plot_map(np.asarray(smooth_mom1), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=["",""], xlabel="", ylabel="", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)    
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")

                    ret = cfp.plot_map(np.asarray(isolated_data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""],xlabel=xlabel, ylabel="", xlim=[-1,1], ylim=[-1,1], aspect_data='equal')  
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)   
                    cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

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
                    PDF_obj.append(K)
                    sigma.append(np.std(moms[1]))
                    #skewness.append(kurtosis(moms[1].flatten(), 's'))
                    kurtosis.append(kurtosis_own(np.asarray(moms[1]).flatten()))

                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # for after turbulence isolation
                    PDF_obj_isolated.append(K)
                    sigma_isolated.append(np.std(isolated_data))
                    #skewness_isolated.append(kurtosis(isolated_data.flatten(), 's'))
                    kurtosis_isolated.append(kurtosis_own(np.asarray(isolated_data).flatten()))
            
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
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Ideal images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0_SE
                        CO_10_SE_mom0.append(moms[0])
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2_SE
                        CO_10_SE_mom2.append(moms[2])
                        moms[imom] = moms[imom].T # transpose for only the 45.0

                    # For Appen. Fig.
                    ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""], xlabel="", ylabel="", xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)                      
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+"_SE.pdf")

                    # Set with individual colorbars (Fig. 1)
                    ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom],xlabel=xlabel, ylabel=ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)          
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+"_cb_SE.pdf")

                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                CO_10_SE_before.append(moms[1])
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append(('SE', file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction

                # for the correction factors map
                CO_10_SE.append(isolated_data)

                all_sigmas.append(('SE', file, np.std(isolated_data)))

                ret = cfp.plot_map(np.asarray(isolated_data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""],xlabel=xlabel, ylabel="",  xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig.
                cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[1], normalised_coords=True)    
                cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated_SE.pdf")

                # Fourier Analysis Data for Synthetic CO - SimEnd
                K_P = fourier_spectrum(isolated_data)
                FTdata_SE.append(K_P)
                
                # Getting the PDF for SimEnd - CO (1-0)
                if get_LOS(file) == 1:
                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1))
                    PDF_obj_SE.append(K)
                    sigma_SE.append(np.std(isolated_data))
                    #skewness_SE_before.append(kurtosis(moms[1].flatten(), 's'))
                    kurtosis_SE_before.append(kurtosis_own(np.asarray(moms[1]).flatten()))
                    #skewness_SE_after.append(kurtosis(isolated_data.flatten(), 's'))
                    kurtosis_SE_after.append(kurtosis_own(np.asarray(isolated_data).flatten()))

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
                    PPV = resize_45(PPV, "J21")
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
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Ideal images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0
                        CO_21_1tff_mom0.append(moms[0])
                        moms[imom] = moms[imom].T # transpose for all
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        if get_LOS(file) == 0 | get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2
                        CO_21_1tff_mom2.append(moms[2])
                        if get_LOS(file) == 1: moms[imom] = moms[imom].T # transpose for only the 45.0
                    
                    if get_LOS(file) == 1:
                        # For Appen. Fig.
                        ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""], xlabel="", ylabel="",  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+".pdf")

                        # Set with individual colorbars (Fig. 1)
                        ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom],xlabel=xlabel, ylabel=ylabel,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                        cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                        cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                        # plotting a common colorbar, only for seismic, universal vmin and vmax
                        cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar.pdf", panels=2)
        
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                CO_21_1tff_before.append(moms[1])
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append((file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction

                # for the correction factors map
                CO_21_1tff.append(isolated_data)

                all_sigmas.append((file, np.std(isolated_data)))
                if get_LOS(file) == 1:
                    # getting the smoothed maps
                    ret = cfp.plot_map(np.asarray(smooth_mom1), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=["",""], xlabel="", ylabel="",  xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")

                    ret = cfp.plot_map(np.asarray(isolated_data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlabel=xlabel, ylabel="", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                    cfp.plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

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
                    PDF_obj.append(K)
                    sigma.append(np.std(moms[1]))
                    #skewness.append(kurtosis(moms[1].flatten(), 's'))
                    kurtosis.append(kurtosis_own(np.asarray(moms[1]).flatten()))

                    K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # after turbulence isolation
                    PDF_obj_isolated.append(K)
                    sigma_isolated.append(np.std(isolated_data))
                    #skewness_isolated.append(kurtosis(isolated_data.flatten(), 's'))
                    kurtosis_isolated.append(kurtosis_own(np.asarray(isolated_data).flatten()))
            
            # Doing the same as above for Data_SimEnd
            files = ["PPV_45.0_J21_SE.npy"] 
            for file in files:
                moms = [] # empty list to store all the moment maps

                # read PPV data and V axis
                PPV = np.load(path+"/Data_SimEnd/"+file) # loading the data
                Vrange = np.load(path+"/Data_SimEnd/"+"Vrange.npy")
                PPV = resize_45(PPV, "J21")
                xlabel = xyzlabels[1] # y-axis on the bottom
                ylabel = xyzlabels[3] # combination of x and z on the vertical

                # loop over moments
                for imom, moment_map in enumerate(moment_maps):
                    # compute moment maps
                    print("Computing moment "+str(imom)+" map...")
                    if imom==0: mom = zero_moment(PPV, Vrange); mom = rescale_data(mom)  # need to rescale the 0th moment map alone
                    if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Ideal images
                    if imom==2: mom = second_moment(PPV, Vrange)
                    moms.append(mom) # append to bigger list of moment maps

                    # plot moment maps, since PPV is used in both Fig.1 and Appen. , we need one set with colorbars and one set without
                    # Set with a common colorbar (Appen. Fig.)
                    if imom == 0: # 0th moment
                        vmin = vmin_0
                        vmax = vmax_0_SE
                        moms[imom] = moms[imom].T # transpose for all
                        CO_21_SE_mom0.append(moms[0])
                    if imom == 1: # 1st moment
                        vmin = vmin_1
                        vmax = vmax_1
                        moms[imom] = moms[imom].T # transpose for all but 90.0
                    if imom == 2: # 2nd moment
                        vmin = vmin_2
                        vmax = vmax_2_SE
                        moms[imom] = moms[imom].T # transpose for only the 45.0
                        CO_21_SE_mom2.append(moms[2])
                
                    # For Appen. Fig.
                    ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar=False, axes_format=["", ""],xlabel="", ylabel="",  xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+".pdf")

                    # Set with individual colorbars (Fig. 1)
                    ret = cfp.plot_map(np.asarray(moms[imom]), cmap=cmaps[imom], vmin=vmin, vmax=vmax, colorbar = True, cmap_label=cmap_labels[imom], xlabel=xlabel, ylabel=ylabel, xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                    cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")
        
                # Smoothing (turbulence isolation) of moment 1
                print("Now doing turbulence isolation on moment 1")
                CO_21_SE_before.append(moms[1])
                smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
                all_sigmas_before.append(('SE', file , np.std(moms[1])))

                # Generating isolated map and then plotting it
                print("Now subtracting turbulence isolated moment 1")
                isolated_data = moms[1] - smooth_mom1 # subtraction

                # for the correction factors map
                CO_21_SE.append(isolated_data)

                all_sigmas.append(('SE', file, np.std(isolated_data)))

                ret = cfp.plot_map(np.asarray(isolated_data), cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlabel=xlabel, ylabel="", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # common colorbar for Appen. Fig. 
                cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=img_names[2], normalised_coords=True)  
                cfp.show_or_save_plot(save=outpath+file[:-4]+"_"+moment_maps[1]+"_isolated.pdf")

                # Fourier Analysis Data for Synthetic CO (2-1) - SimEnd
                K_P = fourier_spectrum(isolated_data)
                FTdata_SE.append(K_P)
                
                # Getting the PDF for SimEnd - CO (2-1) - SimEnd
                K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # only after turbulence isolation
                PDF_obj_SE.append(K)
                sigma_SE.append(np.std(isolated_data))
                #skewness_SE_before.append(kurtosis(moms[1].flatten(), 's'))
                kurtosis_SE_before.append(kurtosis_own(np.asarray(moms[1]).flatten()))
                #skewness_SE_after.append(kurtosis(isolated_data.flatten(), 's'))
                kurtosis_SE_after.append(kurtosis_own(np.asarray(isolated_data).flatten()))

    # Getting the correction factor maps and the PDFs
    ##################### Correction factor maps - MOMENT 1, BEFORE #######################
    correction_CO_10_1tff_before = np.asarray((CO_10_1tff_before[0]-ideal_1tff_before[0])) # 10 at 1tff
    correction_CO_10_SE_before = np.asarray((CO_10_SE_before[0]-ideal_SE_before[0])) # 10 at SE

    correction_CO_21_1tff_before =np.asarray((CO_21_1tff_before[0]-ideal_1tff_before[0])) # 21 at 1tff
    correction_CO_21_SE_before = np.asarray((CO_21_SE_before[0]-ideal_SE_before[0])) # 21 at SE

    corrections_before = [correction_CO_10_1tff_before, correction_CO_10_SE_before, correction_CO_21_1tff_before, correction_CO_21_SE_before]

    # Plotting the correction factor maps
    ret = cfp.plot_map(correction_CO_10_1tff_before, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[4],vmin=-0.3, vmax=0.3, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[0], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_1tff_mom1_before.pdf")

    ret = cfp.plot_map(correction_CO_21_1tff_before, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[5], vmin=-0.3, vmax=0.3, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$",xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[2], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_1tff_mom1_before.pdf")

    ret = cfp.plot_map(correction_CO_10_SE_before, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[4],vmin=-0.3, vmax=0.3, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[1], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_SE_mom1_before.pdf")

    ret = cfp.plot_map(correction_CO_21_SE_before, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[5], vmin=-0.3, vmax=0.3,xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[3], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_SE_mom1_before.pdf")

    # Obtaining correction PDFs
    for i in range(len(corrections_before)):
        K = cfp.get_pdf(corrections_before[i], bins=correction_bins_values) 
        PDF_correction_obj_1_before.append(K)
        correction_sigmas_1_before.append(np.std(corrections_before[i]))
    
    # Plotting correction PDFs
    for i in range(len(PDF_correction_obj_1_after)):
        if i==1 or i==3: alpha=0.5
        else: alpha=1
        if i==0 or i==2: line_color=line_colours[1]
        else: line_color=line_colours[2]
        cfp.plot(x=PDF_correction_obj_1_after[i].bin_edges, y=PDF_correction_obj_1_after[i].pdf, alpha=alpha, type='pdf', label=PDF_img_names_correc(i, correction_sigmas_1_before[i]), color=line_color)
    cfp.plot(xlabel=correction_xlabel, ylabel="PDF", fontsize='small', ylog=True, legend_loc='upper left', save=outpath+"correction_PDF_mom1_before.pdf")

    ##################### Correction factor maps - MOMENT 1, AFTER #######################
    correction_CO_10_1tff = np.asarray((CO_10_1tff[0]-ideal_1tff[0])) # 10 at 1tff
    correction_CO_10_SE = np.asarray((CO_10_SE[0]-ideal_SE[0])) # 10 at SE

    correction_CO_21_1tff = np.asarray((CO_21_1tff[0]-ideal_1tff[0])) # 21 at 1tff
    correction_CO_21_SE = np.asarray((CO_21_SE[0]-ideal_SE[0])) # 21 at SE

    corrections = [correction_CO_10_1tff, correction_CO_10_SE, correction_CO_21_1tff, correction_CO_21_SE]

    # Plotting the correction factor maps
    ret = cfp.plot_map(correction_CO_10_1tff, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[4], xlabel=r'$y$', vmin=-0.3, vmax=0.3,ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[0], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_1tff_mom1_after.pdf")

    ret = cfp.plot_map(correction_CO_21_1tff, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[5], xlabel=r'$y$', vmin=-0.3, vmax=0.3,ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[2], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_1tff_mom1_after.pdf")

    ret = cfp.plot_map(correction_CO_10_SE, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[4], xlabel=r'$y$', vmin=-0.3, vmax=0.3,ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[1], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_SE_mom1_after.pdf")

    ret = cfp.plot_map(correction_CO_21_SE, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[5], xlabel=r'$y$',vmin=-0.3, vmax=0.3, ylabel=r"$\sqrt{x^2 + z^2}$", xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[3], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_SE_mom1_after.pdf")

    # Obtaining correction PDFs
    for i in range(len(corrections)):
        K = cfp.get_pdf(corrections[i], bins=correction_bins_values) 
        PDF_correction_obj_1_after.append(K)
        correction_sigmas_1_after.append(np.std(corrections[i]))
    
    # Plotting correction PDFs
    for i in range(len(PDF_correction_obj_1_after)):
        if i==1 or i==3: alpha=0.5
        else: alpha=1
        if i==0 or i==2: line_color=line_colours[1]
        else: line_color=line_colours[2]
        cfp.plot(x=PDF_correction_obj_1_after[i].bin_edges, y=PDF_correction_obj_1_after[i].pdf, alpha=alpha, type='pdf', label=PDF_img_names_correc(i, correction_sigmas_1_after[i]), color=line_color)
    cfp.plot(xlabel=correction_xlabel, ylabel="PDF", fontsize='small', ylog=True, legend_loc='upper left', save=outpath+"correction_PDF_mom1_after.pdf")

    ###########CORRECTION FACTOR MAPS MOMENT ZERO#######################
    # Getting the correction factor maps and the PDFs
    correction_CO_10_1tff_mom0 = np.asarray((CO_10_1tff_mom0[0])/ideal_1tff_mom0[0]) # 10 at 1tff
    correction_CO_10_SE_mom0 = np.asarray((CO_10_SE_mom0[0])/ideal_SE_mom0[0]) # 10 at SE

    correction_CO_21_1tff_mom0 = np.asarray((CO_21_1tff_mom0[0])/ideal_1tff_mom0[0]) # 21 at 1tff
    correction_CO_21_SE_mom0 = np.asarray((CO_21_SE_mom0[0])/ideal_SE_mom0[0] )# 21 at SE

    corrections_0 = [correction_CO_10_1tff_mom0, correction_CO_10_SE_mom0, correction_CO_21_1tff_mom0, correction_CO_21_SE_mom0]

    # Plotting the correction factor maps
    ret = cfp.plot_map(correction_CO_10_1tff_mom0, cmap=cmaps[4],colorbar=True, cmap_label=correction_cmap_lables[0], log=True, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", vmin=0.1, vmax=10, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[0], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_1tff_mom0.pdf")

    ret = cfp.plot_map(correction_CO_21_1tff_mom0, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[1], log=True, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$",vmin=0.1, vmax=10,  xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[2], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_1tff_mom0.pdf")

    ret = cfp.plot_map(correction_CO_10_SE_mom0, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[0], log=True, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", vmin=0.1, vmax=10, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[1], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_SE_mom0.pdf")

    ret = cfp.plot_map(correction_CO_21_SE_mom0, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[1], log=True, xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", vmin=0.1, vmax=10, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[3], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_SE_mom0.pdf")

    # Obtaining correction PDFs
    for i in range(len(corrections_0)):
        K = cfp.get_pdf(corrections_0[i], bins=correction_bins_values) 
        PDF_correction_obj_0.append(K)
        correction_sigmas_0.append(np.std(corrections[i]))
    
    # Plotting correction PDFs
    for i in range(len(PDF_correction_obj_0)):
        if i==1 or i==3: alpha=0.5
        else: alpha=1
        if i==0 or i==2: line_color=line_colours[1]
        else: line_color=line_colours[2]
        cfp.plot(x=PDF_correction_obj_0[i].bin_edges, y=PDF_correction_obj_0[i].pdf, alpha=alpha, type='pdf', label=PDF_img_names_correc(i, correction_sigmas_0[i]), color=line_color)
    cfp.plot(xlabel=correction_xlabel, ylabel="PDF", fontsize='small', ylog=True, legend_loc='upper left', save=outpath+"correction_PDF_mom0.pdf")

    ###################CORRECTION FACTOR MAPS - MOMENT 2 ######################
    # Getting the correction factor maps and the PDFs
    correction_CO_10_1tff_2 = np.asarray((CO_10_1tff_mom2[0])/ideal_1tff_mom2[0] )# 10 at 1tff
    correction_CO_10_SE_2 = np.asarray((CO_10_SE_mom2[0])/ideal_SE_mom2[0] )# 10 at SE

    correction_CO_21_1tff_2 =np.asarray( (CO_21_1tff_mom2[0])/ideal_1tff_mom2 )# 21 at 1tff
    correction_CO_21_SE_2 =np.asarray( (CO_21_SE_mom2[0])/ideal_SE_mom2[0]) # 21 at SE

    corrections_2 = [correction_CO_10_1tff_2, correction_CO_10_SE_2, correction_CO_21_1tff_2, correction_CO_21_SE_2]

    # Plotting the correction factor maps
    ret = cfp.plot_map(correction_CO_10_1tff_2, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[2], xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", log=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[0], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_1tff_mom2.pdf")

    ret = cfp.plot_map(correction_CO_21_1tff_2, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[3], xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", log=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[2], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_21_1tff_mom2.pdf")

    ret = cfp.plot_map(correction_CO_10_SE_2, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[2], xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", log=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[1], normalised_coords=True)
    cfp.show_or_save_plot(save=outpath+"correction_map_10_SE_mom2.pdf")

    ret = cfp.plot_map(correction_CO_21_SE_2, cmap=cmaps[4], colorbar=True, cmap_label=correction_cmap_lables[3], xlabel=r'$y$', ylabel=r"$\sqrt{x^2 + z^2}$", log=True, xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
    cfp.plot(ax=ret.ax()[0], x=img_names_xpos, y=img_names_ypos, text=correction_labels[3], normalised_coords=True)
    t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
    cfp.show_or_save_plot(save=outpath+"correction_map_21_SE_mom2.pdf")

    # Obtaining correction PDFs
    for i in range(len(corrections_2)):
        K = cfp.get_pdf(corrections_2[i], bins=correction_bins_values) 
        PDF_correction_obj_2.append(K)
        correction_sigmas_2.append(np.std(corrections[i]))
    
    # Plotting correction PDFs
    for i in range(len(PDF_correction_obj_2)):
        if i==1 or i==3: alpha=0.5
        else: alpha=1
        if i==0 or i==2: line_color=line_colours[1]
        else: line_color=line_colours[2]
        cfp.plot(x=PDF_correction_obj_2[i].bin_edges, y=PDF_correction_obj_2[i].pdf, alpha=alpha, type='pdf', label=PDF_img_names_correc(i, correction_sigmas_2[i]), color=line_color)
    cfp.plot(xlabel=correction_xlabel, ylabel="PDF", fontsize='small', ylog=True, legend_loc='upper left', save=outpath+"correction_PDF_mom2.pdf")

    ############POWER SPECTRA#############
    # Plotting the FTs - before isolation, 1tff
    for i in range(len(FTdata_raw)):
        x=FTdata_raw[i][0][1:kmax]; y=FTdata_raw[i][1][1:kmax]
        params = {"a":[1e-4, 1e-2, 1], "n":[-4, -2, -1]}
        fit_values = cfp.fit(func, xdat=x[kmin:], ydat=np.log(y[kmin:]), perr_method='systematic', params=params)
        a=cfp.round(fit_values.popt[0], 2, str_ret=True); n=cfp.round(fit_values.popt[1], 2, str_ret=True)
        cfp.plot(x=x, y=y, label=img_names[i]+FT_slope_labels(fit_values.perr,n), color=line_colours[i])
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle='dotted')
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
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle='dotted')
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
        cfp.plot(x=x[kmin:], y=np.exp(func(x[kmin:], *fit_values.popt)), alpha=0.5, color=line_colours[i], linestyle='dotted')
    secax1 = plt.gca().secondary_xaxis('top', functions=(secax_forward, secax_backward))
    secax1.set_xlabel(r"$\ell\,/\,\mathrm{pc}$")
    secax1.tick_params(axis='x', direction='in', length=0, which = 'minor', top=False, bottom=True)
    cfp.plot(x=img_PDF_names_xpos+0.003, y=img_PDF_names_ypos-0.55, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(legend_loc='lower left', xlabel=FT_xy_labels[0], ylabel=FT_xy_labels[1], fontsize='small', ylog=True,  xlog=True, save=outpath+"FT_after_SE.pdf")

    ####### ALL PDFS #############
    # Plotting the PDFs for before isolation
    for i in range(len(PDF_obj)):
        cfp.plot(x=PDF_obj[i].bin_edges, y=PDF_obj[i].pdf, type='pdf', label=PDF_img_names(i, sigma[i]), color=line_colours[i])
        good_ind = PDF_obj[i].pdf > 0
        fitobj = cfp.fit(gauss_func, PDF_obj[i].bin_center[good_ind], np.log(PDF_obj[i].pdf[good_ind]))
        cfp.plot(x=xfit, y=np.exp(gauss_func(xfit, *fitobj.popt)), alpha=0.5, color=line_colours[i], linestyle='dashed')
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=img_types[0], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"before_isolation_PDF.pdf")

    # Plotting the PDFs for after isolation
    for i in range(len(PDF_obj_isolated)):
        cfp.plot(x=PDF_obj_isolated[i].bin_edges, y=PDF_obj_isolated[i].pdf, type='pdf', label=PDF_img_names(i, sigma_isolated[i]), color=line_colours[i])
        good_ind = PDF_obj_isolated[i].pdf > 0
        fitobj = cfp.fit(gauss_func, PDF_obj_isolated[i].bin_center[good_ind], np.log(PDF_obj_isolated[i].pdf[good_ind]))
        cfp.plot(x=xfit, y=np.exp(gauss_func(xfit, *fitobj.popt)), alpha=0.5, color=line_colours[i], linestyle='dashed')
    cfp.plot(x=img_PDF_names_xpos, y=img_PDF_names_ypos-0.2, text=img_types[1], backgroundcolor="white", fontsize='small', transform=plt.gca().transAxes)
    cfp.plot(xlabel=cmap_labels[1], ylabel="PDF", fontsize='small', ylog=True, xlim=[xmin, xmax], ylim=[ymin,ymax], legend_loc='upper left', save=outpath+"after_isolation_PDF.pdf")

    # Plotting the PDFs for after isolation - FOR SIMEND
    for i in range(len(PDF_obj_SE)):
        cfp.plot(x=PDF_obj_SE[i].bin_edges, y=PDF_obj_SE[i].pdf, type='pdf', label=PDF_img_names(i, sigma_SE[i]), color=line_colours[i])
        good_ind = PDF_obj_SE[i].pdf > 0
        fitobj = cfp.fit(gauss_func, PDF_obj_SE[i].bin_center[good_ind], np.log(PDF_obj_SE[i].pdf[good_ind]))
        cfp.plot(x=xfit, y=np.exp(gauss_func(xfit, *fitobj.popt)), alpha=0.5, color=line_colours[i], linestyle='dashed')
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
    #print("skewness: ", skewness)
    print("kurtosis: ", kurtosis)
    #print("skewness_iso: ", skewness_isolated)
    print("kurtosis_iso: ", kurtosis_isolated)
    #print("skewness_SE_before: ", skewness_SE_before)
    print("kurtosis_SE_before: ", kurtosis_SE_before)
    print("kurtosis_SE_after: ", kurtosis_SE_after)
