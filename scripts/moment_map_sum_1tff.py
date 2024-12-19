#!/usr/bin/env python
# This file plots moment map summary at 1tff (Fig. 1)

import numpy as np
import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import cfpack as cfp
from cfpack import print, stop, hdfio, matplotlibrc
from scipy import stats as st
from astropy import constants as c

from main import *
from all_functions import *
from universal_variables import *

all_moments = moments_45()
fmp = first_moment_plotter()

def mom_map_sum_1tff(): # makes the moment map summary for 1tff and 1.2tff
    fmp.with_colorbar(data=all_moments.first_mom_ideal, text=img_names[0], save=outpath+files_ideal_npy[0][:-4])

mom_map_sum_1tff()