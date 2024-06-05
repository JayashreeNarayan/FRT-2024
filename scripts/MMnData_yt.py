#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
from yt import *                                                          #
import numpy as np                                                        #
from scipy.constants import parsec, m_p                                   #
import os, sys                                                            #
                                                                          #
parsec, m_p, mu = parsec*1e+2, m_p*1e+3, 2.4237981621576                  #
fi0, th0 = 90., 0.                                                        #
level = 4                                                                 #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
pf = load("ChemoMHD_hdf5_plt_cnt_0001")                                   #
                                                                          #
dd=pf.all_data()                                                          #
                                                                          #
dims=pf.domain_dimensions*pf.refine_by**level                             #
                                                                          #
dims=[dims[0], dims[1], dims[2]]                                          #
                                                                          #
pf.force_periodicity()                                                    #
                                                                          #
cube=pf.smoothed_covering_grid(level, left_edge=pf.domain_left_edge, dims=dims)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                          Density                                        #
dens=cube['dens']                                                         #
                                                                          #
dens=np.array(dens)                                                       #
                                                                          #
dens=dens.reshape(dims[0], dims[1], dims[2])                              #
                                                                          #
dens=np.rot90(dens)                                                       #
                                                                          #
dens=dens/m_p/mu                                                          #
                                                                          #
size=dens.shape                                                           #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
#                             velx/y/z                                    #
vx, vy, vz = cube['velx'], cube['vely'], cube['velz']                     #
                                                                          #
vx, vy, vz = np.array(vx), np.array(vy), np.array(vz)                     #
                                                                          #
vx, vy, vz = vx.reshape(dims[0], dims[1], dims[2]), vy.reshape(dims[0], dims[1], dims[2]), vz.reshape(dims[0], dims[1], dims[2])
                                                                          #
vx, vy, vz = np.rot90(vx), np.rot90(vy), np.rot90(vz)                     #
                                                                          #
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -#
if th0 == 90.:                                                            #
	dens = np.moveaxis(dens, 0, 1)                                    #
	vx = np.moveaxis(vx, 0, 1)                                        #
	vy = np.moveaxis(vy, 0, 1)                                        #
	vz = np.moveaxis(vz, 0, 1)                                        #
	vx, vy = vy, vx                                                   #
                                                                          #
cwd = os.getcwd()                                                         #
                                                                          #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
#                    Face-on case                                         #
if fi0 == 0. and th0 == 0. :                                              #
	                                                                  #
	F = np.sum(dens*vz*1e-5, axis = 2)/np.sum(dens, axis = 2)         #
	                                                                  #
	S = np.sum(dens*(vz*1e-5)**2, axis = 2)/np.sum(dens, axis = 2)    #
	                                                                  #
	S = np.sqrt(S - F**2)                                             #
	                                                                  #
	#np.save("{}/FMM_{}_{}".format(cwd, fi0, th0), F)                 #
	                                                                  #
	#np.save("{}/SMM_{}_{}".format(cwd, fi0, th0), S)                 #
	                                                                  #
elif fi0 == 90. and (th0 == 0. or th0 == 90.):                            #
	                                                                  #
	F = np.sum(dens*vx*1e-5, axis = 1)/np.sum(dens, axis = 1)         #
	                                                                  #
	S = np.sum(dens*(vx*1e-5)**2, axis = 1)/np.sum(dens, axis = 1)    #
	                                                                  #
	S = np.sqrt(S - F**2)                                             #
	                                                                  #
	#np.save("{}/FMM_{}_{}".format(cwd, fi0, th0), F)                 #
	                                                                  #
	#np.save("{}/SMM_{}_{}".format(cwd, fi0, th0), S)                 #
	                                                                  #
else:                                                                     #
	                                                                  #
	pass                                                              #
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-#
