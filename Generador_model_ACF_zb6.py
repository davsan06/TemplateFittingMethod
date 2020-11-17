#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:01:54 2020

@author: davsan06
"""

import camb
import numpy as np
import matplotlib.pyplot as plt
import os
# import scipy.integrate as sci
import scipy.special as spe
from astropy.table import Table
from scipy.interpolate import interp1d
# import seaborn as sns
import datetime
# from iminuit import Minuit
#import multiprocessing as mp

path='/scratch/davsan06/maglim_BAO'
path_acfs_mocks = os.path.join(path, 'test_binang_1.0_10.0_nbins_50')
path_model = os.path.join(path, 'acf_model')
path_mocks = os.path.join(path, 'acfs_maglim')
path_martin = '/scratch/monroy/cosmosis/cosmosis/maglim_v2.2_newzbinning_jointmask'

###############################################################################
#######                         COSMOLOGY                             #########
###############################################################################
# Set custom cosmology (the one inserted to generate mocks)
pars = camb.CAMBparams(H0=69.0, 
                       ombh2=0.048*0.69**2, 
                       omch2=0.252*0.69**2,
                       omk=0,
                       tau=0.08,
                       omnuh2=0.00083,
                       YHe=0.245341)
pars.InitPower.set_params(As=2.19e-9, ns=0.97)
pars.DarkEnergy.set_params(w=-1.00, wa=0.00)

# Reduced Hubble constant
h=pars.H0/100

###############################################################################
#######                DISTANCE BETWEEN GALAXIES                      #########
###############################################################################
results = camb.get_results(pars)

def distance_btw_gal(z1, z2, theta):
    # INPUT
    # zi : redshift to object i
    # theta : angle in degrees btween objects
    # OUPUT
    # s: distance in Mpc
    
    s1 = results.comoving_radial_distance(z=z1)
    s2 = results.comoving_radial_distance(z=z2)

    s = np.sqrt(s1**2 + s2**2 - 2*s1*s2*np.cos(np.radians(theta)))

    return(s)
    
###############################################################################
#######                     Z-SPACE CORR FUNCT.                       #########
###############################################################################
b = 1.0 # Fixed bias, we dont care about amplitud, just shape info

def xi_zspace(z1, z2, theta, b):
    # z1=z_inter[25]
    # z2=z_inter[25]
    # theta = 4.0
    
    # Getting index
    ind = np.where(np.isclose(z, (z1+z2)/2, rtol=5e-02, atol=5e-02))[0][0]
    
    s = distance_btw_gal(z1=z1, z2=z2, theta=theta)
    
    xi = b**2/(2*np.pi**2)*np.trapz(k**2*spe.spherical_jn(0, k*s)*pk[ind], k)
    return(xi)
    
#plt.semilogx(k,k**2*spe.spherical_jn(0, k*s)*pk[ind])
###############################################################################
#######                        ACF = ACF(THETA)                       #########
###############################################################################
def acf(theta):
    
    # Generate Xi(z_1, z_2) matrix
    xi_mat = np.zeros((len(z_inter),len(z_inter)))

    i=0
    for zi in z_inter:
        j=0
        for zj in z_inter:
        
            xi= xi_zspace(z1=zi, z2=zj, theta=theta, b=b)
        
            xi_mat[i,j] = xi
        
            j = j+1
        i=i+1
        print('z-progress: ',round(i/len(z_inter)*100,2), '%')
    
    w = np.trapz(phi_interp(z_inter)*np.trapz(xi_mat*phi_interp(z_inter), z_inter), z_inter)
    return(w)
    
def acf_wrapper(theta_array):    
    
    acf_array = np.array([])  

    l=0
    for th in theta_array:
        l=l+1
        print('Angular progress: ', round(l/len(theta_array)*100, 2), '% \n')
        acf_array = np.append(acf_array, acf(theta=th))
    
    return(acf_array)   
###############################################################################
#######                     SELECTION FUNCTION                        #########        
###############################################################################
cat = Table.read(os.path.join(path_martin, 'nz_mag_lim_lens_sample_combined_jointmask_z_mc.fits'))
cat.info

z_phi = cat['Z_LOW']
z_array = np.linspace(min(z_phi), max(z_phi), 1000)

# Angular binning
theta_array = np.linspace(0.0, 13.0, 70)

# Hiperparameters
kmax=10**2

z_label = [0, [0.2, 0.4], [0.4, 0.55], [0.55, 0.7], [0.7, 0.85], [0.85, 0.95], [0.95, 1.05]]

# Loop over all z-bins
# for z_bin in np.arange(1,7,1):
    # z_bin = np.arange(1,7,1)[0]
        
z_bin=6
   
print('z-bin : ', z_bin)
    
t1 =  datetime.datetime.now()
    
# Loading selection function
phi = cat['BIN{}'.format(z_bin)]
phi_interp = interp1d(z_phi, phi, kind='linear')
    
###############################################################################
#######                     MATTER POWER SPECTRA                      #########
###############################################################################
z_mean = np.trapz(z_array*phi_interp(z_array), z_array)
z_inter = np.linspace(z_mean-0.29, z_mean+0.3, 100) # TO CHECK: optimum width per z-bin

pars.set_matter_power(redshifts=z_inter, kmax=kmax) # Evolution of Pk with z, no growth factor needed in w(theta) eq.
results = camb.get_results(pars)
k, z, pk = results.get_matter_power_spectrum(minkh=1e-5/h, maxkh=kmax/h, npoints = 5*10**5) # Good Pk, much time computing w(theta) integral
k = k*h
pk = pk/h**3
    
#plt.loglog(k,pk[5])

#plt.loglog(k,pk[14])        
acf_array = acf_wrapper(theta_array)
    
# Saving theta-ACF arrays per z-bin
np.savetxt(os.path.join(path_model, 'theta_acf_maglim_zbin_{}.txt'.format(z_bin)), np.array([theta_array,acf_array]))
# Saving plot
plt.plot(theta_array, theta_array**2*acf_array)
plt.title('MagLim ACF z-bin{}: {} < z < {} \n'.format(z_bin, z_label[z_bin][0], z_label[z_bin][1]), fontsize=18)        
plt.xlabel('$\Theta$ (deg)', fontsize=16)
plt.ylabel('$\Theta^2 \cdot w(\Theta)$', fontsize=16)
plt.savefig(os.path.join(path_model, 'acf_fig_maglim_zbin_{}.png'.format(z_bin)),
            format='png',
            dpi=150,
            bbox_inches='tight')
plt.show()
        
t2 =  datetime.datetime.now()
print(t2-t1)
print("Ejecucion del z-bin {} finalizada en {}".format(z_bin, t2-t1))
    

