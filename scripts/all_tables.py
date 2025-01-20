#!/usr/bin/env python
# This file plots all the correction maps for 1tff (Fig 9 to 12)

import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
from scipy import stats as st
from astropy import constants as c
import os

from util_scripts.main import *
from util_scripts.all_functions import *
from util_scripts.universal_variables import *

moments_45_ = moments_45()
all_moments_ = moments()

def kurtosis_values():
    path='../all_table_tex_files/' # save these files to a different subfolder
    if not os.path.isdir(path):
        cfp.run_shell_command('mkdir '+path)

    idealised_otff_before = moments_45_.first_mom_ideal_kurt[0]
    idealised_otff_after = moments_45_.first_mom_isolated_ideal_kurt[0]
    idealised_ottff_before = moments_45_.first_mom_ideal_kurt[1]
    idealised_ottff_after = moments_45_.first_mom_isolated_ideal_kurt[1]

    cooz_otff_before = moments_45_.first_mom_co10_kurt[0]
    cooz_otff_after = moments_45_.first_mom_isolated_co10_kurt[0]
    cooz_ottff_before = moments_45_.first_mom_co10_kurt[1]
    cooz_ottff_after = moments_45_.first_mom_isolated_co10_kurt[1]

    coto_otff_before = moments_45_.first_mom_co21_kurt[0]
    coto_otff_after = moments_45_.first_mom_isolated_co21_kurt[0]
    coto_ottff_before = moments_45_.first_mom_co21_kurt[1]
    coto_ottff_after = moments_45_.first_mom_isolated_co21_kurt[1]

    # Define the LaTeX table as a formatted string
    latex_code = rf"""
\begin{{table*}}
\caption{{Table of all the kurtosis values for before turbulence isolation and after turbulence isolation at $t=\otff$ and $t=\ottff$ for the first-moment maps of the 3 cases under consideration.}}
\makebox[\textwidth][c]{{
    \begin{{tabular}}{{ccccc}}
\toprule
    & \multicolumn{{2}}{{c}}{{$\otff$}} & \multicolumn{{2}}{{c}}{{$\ottff$}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
& \multicolumn{{1}}{{c}}{{Before isolation}} & \multicolumn{{1}}{{c}}{{After isolation}} & \multicolumn{{1}}{{c}}{{Before isolation}} & \multicolumn{{1}}{{c}}{{After isolation}} \\
\cmidrule(lr){{2-3}}\cmidrule(lr){{4-5}}
Idealised & {idealised_otff_before} & {idealised_otff_after} & {idealised_ottff_before} & {idealised_ottff_after} \\
$\cooz$ & {cooz_otff_before} & {cooz_otff_after} & {cooz_ottff_before} & {cooz_ottff_after} \\
$\coto$ & {coto_otff_before} & {coto_otff_after} & {coto_ottff_before} & {coto_ottff_after} \\
\bottomrule
\end{{tabular}}
}}
\label{{table:pdf_kurtosis}}
\end{{table*}}
"""

    # Write the LaTeX code to a file
    output_file = "kurtosis_table.tex"
    with open(path+output_file, "w") as file:
        file.write(latex_code)

    print(f"LaTeX code written to {path+output_file}")

kurtosis_values()

def generate_sigvod_table(output_filename="table_sigvod.tex"):
    path='../all_table_tex_files/' # save these files to a different subfolder
    if not os.path.isdir(path):
        cfp.run_shell_command('mkdir '+path)
        
    # SigV values for different LOS
    sigv_values = {
        "(1 0 0)": {"idealised_before": all_moments_.first_mom_ideal_sigma[0], "idealised_after": all_moments_.first_mom_isolated_ideal_sigma[0],
                    "cooz_before": all_moments_.first_mom_co10_sigma[0], "cooz_after": all_moments_.first_mom_isolated_co10_sigma[0],
                    "coto_before": all_moments_.first_mom_co21_sigma[0], "coto_after": all_moments_.first_mom_isolated_co21_sigma[0]},
        r"$\frac{1}{\sqrt{2}}$(1 0 1)": {"idealised_before": moments_45_.first_mom_ideal_sigma[0], "idealised_after": moments_45_.first_mom_isolated_ideal_sigma[0],
                    "cooz_before": moments_45_.first_mom_co10_sigma[0], "cooz_after": moments_45_.first_mom_isolated_co10_sigma[0],
                    "coto_before": moments_45_.first_mom_co21_sigma[0], "coto_after": moments_45_.first_mom_isolated_co21_sigma[0]},
        "(0 0 1)": {"idealised_before": all_moments_.first_mom_ideal_sigma[2], "idealised_after": all_moments_.first_mom_isolated_ideal_sigma[2],
                    "cooz_before": all_moments_.first_mom_co10_sigma[2], "cooz_after": all_moments_.first_mom_isolated_co10_sigma[2],
                    "coto_before": all_moments_.first_mom_co21_sigma[2], "coto_after": all_moments_.first_mom_isolated_co21_sigma[2]},
        r"$\frac{1}{\sqrt{2}}$(1 0 1)$_\mathrm{t=\ottff}$": {"idealised_before": all_moments_.first_mom_ideal_sigma[3], "idealised_after": all_moments_.first_mom_isolated_ideal_sigma[3],
                    "cooz_before": all_moments_.first_mom_co10_sigma[3], "cooz_after": all_moments_.first_mom_isolated_co10_sigma[3],
                    "coto_before": all_moments_.first_mom_co21_sigma[3], "coto_after": all_moments_.first_mom_isolated_co21_sigma[3]},
    }

    # Generate table content
    table = r"""
\begin{table*}
\caption{Summary of $\sigvod$ measurements for the three cases considered, before and after turbulence isolation, and for 3~different lines of sight. The factor $f_{\mathrm{rel}}$ is the ratio between the case being considered in the table and the idealised counterpart. The second row is our main line of sight and the last row is our main line of sight at $t=\ottff$.}
\makebox[\textwidth][c]{
    \begin{tabular}{ccccccccccccc}
\toprule
    & \multicolumn{4}{c}{idealised} & \multicolumn{4}{c}{$\cooz$} & \multicolumn{4}{c}{$\coto$} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}
& \multicolumn{2}{c}{Before isolation} & \multicolumn{2}{c}{After isolation} & \multicolumn{2}{c}{Before isolation} & \multicolumn{2}{c}{After isolation} & \multicolumn{2}{c}{Before isolation} & \multicolumn{2}{c}{After isolation}\\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}\cmidrule(lr){12-13}
LOS & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ & $\sigv \left[\,\kmps\right]$ & $f_\mathrm{rel}$ \\ \midrule
"""

    # Build table rows
    for los, values in sigv_values.items():
        f_rel_cooz_before = values["cooz_before"] / values["idealised_before"]
        f_rel_cooz_after = values["cooz_after"] / values["idealised_after"]
        f_rel_coto_before = values["coto_before"] / values["idealised_before"]
        f_rel_coto_after = values["coto_after"] / values["idealised_after"]

        table += fr"{los} & {values['idealised_before']:.2f} & 1.0 " \
                 fr"& {values['idealised_after']:.2f} & 1.0 " \
                 fr"& {values['cooz_before']:.2f} & {f_rel_cooz_before:.3f} " \
                 fr"& {values['cooz_after']:.2f} & {f_rel_cooz_after:.3f} " \
                 fr"& {values['coto_before']:.2f} & {f_rel_coto_before:.3f} " \
                 fr"& {values['coto_after']:.2f} & {f_rel_coto_after:.3f} \\ "

    # Add table footer
    table += r"""
\bottomrule
\end{tabular}
}
\end{table*}
"""

    # Write to a .tex file
    with open(path+output_filename, "w") as file:
        file.write(table)

    print(f"Table successfully written to {path+output_filename}.")

# Call the function
generate_sigvod_table()
