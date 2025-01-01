import numpy as np
import os
import cfpack as cfp

path = "../Data/"
outpath = "../plots/" 
if not os.path.isdir(outpath):
        cfp.run_shell_command('mkdir '+outpath)

Vrange = np.load(path+'Data_1tff/Vrange.npy')
files_ideal_npy = ["FMM_00.0_0.0.npy", "FMM_45.0_0.0.npy", "FMM_90.0_0.0.npy", # 1tff
                    "SMM_00.0_0.0.npy", "SMM_45.0_0.0.npy", "SMM_90.0_0.0.npy", # 1tff
                    "ZMM_00.0_0.0.npy", "ZMM_45.0_0.0.npy", "ZMM_90.0_0.0.npy"] # 1tff

files_ideal_SE_npy = ["FMM_45.0_SE.npy", "SMM_45.0_SE.npy", "ZMM_45.0_SE.npy"] # SE

files_co10_npy = ["PPV_00.0_0.npy", "PPV_45.0_0.npy", "PPV_90.0_0.npy"] # 1tff

file_co10_SE_npy = ["PPV_45.0.npy"] # SE

files_co21_npy = ["PPV_00.0_0_J21.npy", "PPV_45.0_0_J21.npy", "PPV_90.0_0_J21.npy"] # 1tff

file_co21_SE_npy = ["PPV_45.0_J21_SE.npy"] # SE

files_ideal={}
for file in files_ideal_npy:
    files_ideal[str(file)]= np.load(path+"Data_1tff/Othin/"+file)

for file in files_ideal_SE_npy:
    files_ideal[str(file)]= np.load(path+"Data_SimEnd/Othin/"+file)

files_co10={}
for file in files_co10_npy:
    files_co10[str(file)]= np.load(path+"Data_1tff/"+file)

for file in file_co10_SE_npy:
    files_co10[str(file)]= np.load(path+"Data_SimEnd/"+file)

files_co21={}
for file in files_co21_npy:
    files_co21[str(file)]= np.load(path+"Data_1tff/J21/"+file)

for file in file_co21_SE_npy:
    files_co21[str(file)]= np.load(path+"Data_SimEnd/"+file)

# defining the min and max of the maps universally so that all of them can be compared
vmin_0 = 0. # zeroth moment map
vmax_0 = 4
vmin_1 = -0.55 # 1st
vmax_1 = 0.55
vmin_2 = 0. # 2nd
vmax_2 = 0.6

# LOS labels positions
LOS_labels_xpos = 0.75
LOS_labels_ypos = 0.82

# image title positions
img_names_xpos = 0.05
img_names_ypos = 0.9
img_PDF_names_xpos = 0.02
img_PDF_names_ypos = 0.65

img_names_xpos_cb = 0.05
img_names_ypos_cb = 0.85

# For the PDFs
xmin=-0.45
xmax=+0.45
ymin=1.e-2
ymax=5.e2

linestyle = ['dotted', 'dashed', 'dashdot', 'loosely dotted']
line_colours = ['black', 'magenta', 'blue']
moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
cmaps = ['plasma', 'seismic', 'viridis', 'PuOr']
cmap_labels = [r"${I/\langle I \rangle}$", r"${{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", r"${\sigma_{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", "Correction factor values"]
LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
xyzlabels = [r"$x$", r"$y$", r"$z$", r"$\sqrt{x^2 + z^2}$"]
img_names = ["Ideal", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
img_types=['Before turbulence isolation', 'After turbulence isolation']
FT_xy_labels = [r"$k$", r"$P_\mathrm{tot}$"]

correction_cmap_lables =[
    r"$\left( \frac{I}{\langle I \rangle} \right)_{\mathrm{CO}} / \left( \frac{I}{\langle I \rangle} \right)_{\mathrm{Ideal}}$",
    r"${{v_{\mathrm{LOS,\ CO}}}} - {{v_{\mathrm{LOS,\ Ideal}}}~(\mathrm{km\,s^{-1}})}$",
    r"${\sigma_{v_{\mathrm{LOS,\ CO}}}}/ {\sigma_{v_{\mathrm{LOS,\ Ideal}}}}$"]
correction_labels = [r"$\mathrm{CO}\,(1-0)$", r"$\mathrm{CO}\,(2-1)$"]