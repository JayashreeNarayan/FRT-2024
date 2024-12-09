import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
import argparse
import os
from scipy import stats as st
from astropy import constants as c
from all_functions import *
from universal_variables import *


def pdf_ft_tab1():
    find_plots()

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
        cfp.plot(x=PDF_obj_SE[i].bin_edges, y=PDF_obj_isolated[i].pdf, type='pdf', label=PDF_img_names(i, sigma_SE[i]), color=line_colours[i])
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
