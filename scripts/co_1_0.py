import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *
from universal_variables import *

def co10(correction='N', pdf_ft_tab1='N'):
    find_plots()

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
            if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Idealised images
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
                t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[1] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_map+"_cb.pdf")

                # plotting a common colorbar, only for seismic, universal vmin and vmax
                cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p1.pdf", panels=1) 
                cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p2.pdf", panels=2)
                cfp.plot_colorbar(cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, label=cmap_labels[1], save=outpath+cmaps[1]+"_colorbar_p3.pdf", panels=3)
                cfp.plot_colorbar(cmap=cmaps[3], vmin=vmin_correc, vmax=vmax_correc, symlog=True, label=cmap_labels[3], save=outpath+cmaps[3]+"_colorbar_p2.pdf", panels=2) # grey color panel

        # Smoothing (turbulence isolation) of moment 1
        print("Now doing turbulence isolation on moment 1")
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
            cfp.plot_map(smooth_mom1, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=["",""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') 
            t = plt.text(img_names_xpos, img_names_ypos, img_names[1] , transform=plt.gca().transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
            cfp.plot(xlabel="", ylabel="", save=outpath+file[:-4]+"_"+moment_maps[1]+"_smooth.pdf")

            cfp.plot_map(isolated_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False , axes_format=[None,""], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')  
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
            PDF_obj.append(K)
            sigma.append(np.std(moms[1]))
            skewness.append(skewness_kurtosis(moms[1].flatten(), 's'))
            kurtosis.append(skewness_kurtosis(moms[1].flatten(), 'k'))

            K = cfp.get_pdf(isolated_data, range=(-0.1,+0.1)) # for after turbulence isolation
            PDF_obj_isolated.append(K)
            sigma_isolated.append(np.std(isolated_data))
            skewness_isolated.append(skewness_kurtosis(isolated_data.flatten(), 's'))
            kurtosis_isolated.append(skewness_kurtosis(isolated_data.flatten(), 'k'))

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
            if imom==1: mom = -first_moment(PPV, Vrange) # inverting the image to make it match with the Idealised images
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
            t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[1] , transform=plt.gca().transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
            cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_"+moment_map+"_cb_SE.pdf")

        # Smoothing (turbulence isolation) of moment 1
        print("Now doing turbulence isolation on moment 1")
        smooth_mom1 = smoothing(moms[1]) # Gaussian smoothing for moment 1
        all_sigmas_before.append(('SE', file , np.std(moms[1])))

        # Generating isolated map and then plotting it
        print("Now subtracting turbulence isolated moment 1")
        isolated_data = moms[1] - smooth_mom1 # subtraction

        # for the correction factors map
        CO_10_SE.append(isolated_data)

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
            PDF_obj_SE.append(K)
            sigma_SE.append(np.std(isolated_data))
            skewness_SE_before.append(skewness_kurtosis(moms[1].flatten(), 's'))
            kurtosis_SE_before.append(skewness_kurtosis(moms[1].flatten(), 'k'))
            skewness_SE_after.append(skewness_kurtosis(isolated_data.flatten(), 's'))
            kurtosis_SE_after.append(skewness_kurtosis(isolated_data.flatten(), 'k'))

    if correction == 'Y': 
        return CO_10_SE, CO_10_1tff

    if pdf_ft_tab1 == 'Y': 
        return skewness, skewness_isolated, skewness_kurtosis