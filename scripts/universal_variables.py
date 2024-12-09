path = "../Data/"
outpath = "../plots/" 

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
img_names_ypos = 0.85

img_names_xpos_cb = -19.2
img_names_ypos_cb = 0.85

linestyle = ['dotted', 'dashed', 'dashdot', 'loosely dotted']
line_colours = ['black', 'magenta', 'blue']
moment_maps = ["mom0", "mom1", "mom2"] # Data for all moment maps
cmaps = ['plasma', 'seismic', 'viridis', 'gray']
cmap_labels = [r"${I/\langle I \rangle}$", r"${{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", r"${\sigma_{v_{\mathrm{LOS}}}~(\mathrm{km\,s^{-1}})}$", "Correction factor values"]
LOS_labels = [r"$\left(\begin{array}{c} 0 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 1 \end{array}\right) $", r"$\left(\begin{array}{c} 1 \\ 0 \\ 0 \end{array}\right) $"]
xyzlabels = [r"$x$", r"$y$", r"$z$", r"$\sqrt{x^2 + z^2}$"]
img_names = ["Idealised", "Synthetic CO (1-0)", "Synthetic CO (2-1)"]
img_types=['Before turbulence isolation', 'After turbulence isolation']
