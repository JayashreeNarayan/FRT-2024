import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *
from universal_variables import *

def othin(correction='N', pdf_ft_tab1='N'):
    find_plots()

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
                t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
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
                t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

            else: # all the first moment maps - for Appen. fig
                cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
                t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-3]+"pdf")

                # Also plotting the same with colorbars for Fig. 1
                cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
                t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
                cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'
            
        # Smoothing of the Idealised moment maps - done only for moment 1 maps, skipping moment 2 data
        if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
            smooth_data = smoothing(data)
            all_sigmas_before.append((file , np.std(data)))
            
            # Turbulence isolation of the smoothed data
            isolated_data_othin = data - smooth_data
            all_sigmas.append((file , np.std(isolated_data_othin)))

            # for the correction factors map
            ideal_1tff.append(isolated_data_othin)

            if get_LOS(file) == 1:
                #producing the smoothed maps
                cfp.plot_map(smooth_data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
                t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) 
                t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
                cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-4]+"_smooth.pdf")

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

            # Fourier Analysis Data for Idealised maps
            if get_LOS(file) == 1:
                K_P = fourier_spectrum(isolated_data_othin) # for the turbulence isolated data
                FTdata.append(K_P)
                K_P_raw = fourier_spectrum(data) # for the unisolated data
                FTdata_raw.append(K_P_raw)

                # for the PDFs
                K = cfp.get_pdf(data, range=(-0.1, 0.1)) # without isolation
                PDF_obj.append(K)
                sigma.append(np.std(data))
                skewness.append(skewness_kurtosis(data.flatten(), 's'))
                kurtosis.append(skewness_kurtosis(data.flatten(), 'k'))

                K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1)) # with isolation
                PDF_obj_isolated.append(K)
                sigma_isolated.append(np.std(isolated_data_othin))
                skewness_isolated.append(skewness_kurtosis(isolated_data_othin.flatten(), 's'))
                kurtosis_isolated.append(skewness_kurtosis(isolated_data_othin.flatten(), 'k'))

    # Generating graphs for For SimEnd time 
    files = ["FMM_45.0_SE.npy", "SMM_45.0_SE.npy", "ZMM_45.0_SE.npy"]
    for file in files:
        data = np.load(path+"/Data_SimEnd/Othin/"+file)
        data = resize_45(data, "2D")
        xlabel = xyzlabels[1] # y-axis on the bottom 
        ylabel = xyzlabels[3] # combination of x and z on the vertical

        if file[:1] == "S": # Since the 2nd moment map needs different plot variables
            cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2_SE, colorbar = False, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # colorbar needed since this is for Fig.1
            t = plt.text(img_names_xpos, img_names_ypos, img_names[0], transform=plt.gca().transAxes) # for img_name
            t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
            cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+".pdf")

            # plotting the same with individual colorbars for Fig. 1 eqv.
            cfp.plot_map(data, cmap=cmaps[2], vmin=vmin_2, vmax=vmax_2_SE, colorbar = True, cmap_label=cmap_labels[2], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
            t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
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
            t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
            cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'

        else: # all the first moment maps - for Appen. fig
            cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=["",None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal') # Done for the 1st moment maps separately, needed for Appen. Fig. 
            t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
            cfp.plot(xlabel="", ylabel=ylabel, save=outpath+file[:-4]+".pdf")

            # Also plotting the same with colorbars for Fig. 1
            cfp.plot_map(data, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar = True, cmap_label=cmap_labels[1], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
            t = plt.text(img_names_xpos_cb, img_names_ypos_cb, img_names[0] , transform=plt.gca().transAxes)
            t.set_bbox(dict(facecolor='white', alpha=0.5, linewidth=0))
            cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_cb.pdf") # cb = 'colorbar'
            
        # Smoothing of the Idealised moment maps - done only for moment 1 maps, skipping moment 2 data
        if file[:1] == "F": # 2nd and 0th moment map is not to be smoothed or Turbulence isolated 
            smooth_data = smoothing(data)
            all_sigmas_before.append(('SE', file , np.std(data)))
            # Turbulence isolation of the smoothed data
            isolated_data_othin = data - smooth_data

            # for the correction factors map
            ideal_SE.append(isolated_data_othin)

            all_sigmas.append(('SE', file , np.std(isolated_data_othin)))
            cfp.plot_map(isolated_data_othin, cmap=cmaps[1], vmin=vmin_1, vmax=vmax_1, colorbar=False, axes_format=[None,None], xlim=[-1,1], ylim=[-1,1], aspect_data='equal')
            t = plt.text(img_names_xpos, img_names_ypos, img_names[0] , transform=plt.gca().transAxes) 
            t.set_bbox(dict(facecolor='white', alpha=0.3, linewidth=0))
            cfp.plot(xlabel=xlabel, ylabel=ylabel, save=outpath+file[:-4]+"_isolated.pdf")

            # Fourier Analysis Data for Idealised maps
            K_P = fourier_spectrum(isolated_data_othin)
            FTdata_SE.append(K_P)
            
            # PDFs for after corection alone - SimEnd Case
            if get_LOS(file) == 1:
                K = cfp.get_pdf(isolated_data_othin, range=(-0.1,+0.1))
                PDF_obj_SE.append(K)
                sigma_SE.append(np.std(isolated_data_othin))
                skewness_SE_before.append(skewness_kurtosis(data.flatten(), 's')) # before isolation for SE
                kurtosis_SE_before.append(skewness_kurtosis(data.flatten(), 'k')) # after isolation for SE
                skewness_SE_after.append(skewness_kurtosis(isolated_data_othin.flatten(), 's'))
                kurtosis_SE_after.append(skewness_kurtosis(isolated_data_othin.flatten(), 'k'))
    
    if correction == 'Y': 
        return ideal_1tff, ideal_SE

    if pdf_ft_tab1 == 'Y': 
        return skewness, skewness_isolated, skewness_kurtosis