#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:01:54 2020

@author: davsan06
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import datetime
from scipy.optimize import curve_fit
import scipy.stats as stats

# from scipy.optimize import minimize

path='/scratch/davsan06/Doctorado/maglim_BAO'
path_acfs_mocks = os.path.join(path, 'test_binang_1.0_10.0_nbins_50')
path_model = os.path.join(path, 'acf_model')
path_mocks = os.path.join(path, 'acfs_maglim')
    
###############################################################################
#######                     TEMPLATE METHOD                           #########        
###############################################################################
def template_generator(theta_array,
                       alpha,
                       B_0,B_1,B_2,B_3,B_4,B_5,
                       A0_0,A0_1,A0_2,A0_3,A0_4,A0_5,
                       A1_0,A1_1,A1_2,A1_3,A1_4,A1_5,
                       A2_0,A2_1,A2_2,A2_3,A2_4,A2_5):
    # alpha=1.01
    # B_array = np.ones(6)
    # A0_array = np.ones(6)
    # A1_array = np.ones(6)
    # A2_array = np.ones(6)
    
    B_array=np.array([B_0,B_1,B_2,B_3,B_4,B_5])
    A0_array=np.array([A0_0,A0_1,A0_2,A0_3,A0_4,A0_5])
    A1_array=np.array([A1_0,A1_1,A1_2,A1_3,A1_4,A1_5])
    A2_array=np.array([A2_0,A2_1,A2_2,A2_3,A2_4,A2_5])
    
    template_array=np.array([])
    
    for z_bin in np.arange(1,7):
        #z_bin=1
        theta_model, acf_model=np.loadtxt(os.path.join(path_model, 'theta_acf_maglim_zbin_{}.txt'.format(int(z_bin))))[0], np.loadtxt(os.path.join(path_model, 'theta_acf_maglim_zbin_{}.txt'.format(int(z_bin))))[1]
        
        #  Sin Extrpolar!
        acf_model = interp1d(theta_model, acf_model)   
    
        template = B_array[z_bin-1]*acf_model(alpha*theta_array) + A0_array[z_bin-1]*np.ones_like(theta_array) + np.divide(A1_array[z_bin-1], theta_array) + np.divide(A2_array[z_bin-1], theta_array**2)
        template_array=np.append(template_array,template)
        
    return(template_array)
    
def read_acf_mocks(mock_num):
    # mock_num = 0
    
    mock_array=np.array([])

    for z_bin in np.arange(1,7):
        #z_bin=1
        acf_mock = np.load(os.path.join(path_mocks, 'w_pix_noweights_mock_{}.npy'.format(mock_num)),
                           allow_pickle=True,
                           encoding='bytes').ravel()[0][(z_bin-1, z_bin-1)]
        mock_array=np.append(mock_array,acf_mock)
        
    return(mock_array)
    
###############################################################################
#######                     CHI-2 TO MINIMIZE                         #########        
###############################################################################
def chi_2(theta_array,mock_num,alpha,B_array,A0_array,A1_array,A2_array):
    # alpha=1.00
    # B_1, B_2, B_3, B_4, B_5, B_6 = 3*np.ones(6)
    # A0_1, A0_2, A0_3, A0_4, A0_5, A0_6 = np.zeros(6)
    # A1_1, A1_2, A1_3, A1_4, A1_5, A1_6=-5e-4*np.ones(6)
    # A2_1, A2_2, A2_3, A2_4, A2_5, A2_6=np.zeros(6)
    
    B_0,B_1,B_2,B_3,B_4,B_5=B_array
    A0_0,A0_1,A0_2,A0_3,A0_4,A0_5=A0_array
    A1_0,A1_1,A1_2,A1_3,A1_4,A1_5=A1_array
    A2_0,A2_1,A2_2,A2_3,A2_4,A2_5=A2_array

    temp=template_generator(theta_array=theta_array,
                       alpha=alpha,
                       B_0=B_0,B_1=B_1,B_2=B_2,B_3=B_3,B_4=B_4,B_5=B_5,
                       A0_0=A0_0,A0_1=A0_1,A0_2=A0_2,A0_3=A0_3,A0_4=A0_4,A0_5=A0_5,
                       A1_0=A1_0,A1_1=A1_1,A1_2=A1_2,A1_3=A1_3,A1_4=A1_4,A1_5=A1_5,
                       A2_0=A2_0,A2_1=A2_1,A2_2=A2_2,A2_3=A2_3,A2_4=A2_4,A2_5=A2_5)
    mock=read_acf_mocks(mock_num=mock_num)

    residuo=temp-mock
    chi2=np.dot(residuo,np.dot(np.linalg.inv(cov_mat),residuo.T))
    return(chi2)
###############################################################################
#######                             MAIN                              #########                                 
###############################################################################

# Covariance matriz of the mock data    
cov_mat = np.loadtxt(os.path.join(path_acfs_mocks, 'covmat_maglim.dat'))

# Angular binning 
theta_array = np.loadtxt(os.path.join(path_acfs_mocks, 'theta_acf_bin_0'))[:,0]
   
# Looping over mock set
   
alpha_array = np.array([])
alpha_error_array = np.array([])
chi_2_array = np.array([])
         
l=0
tinicial =  datetime.datetime.now()

#Initial guess
p0=np.array([1,
            2.033356e+00,3.543916e+00,3.240174e+00,3.450081e+00,5.758273e+00,5.084836e+00,
            -8.500000e-05, -1.000000e-04, -7.300000e-05, -4.300000e-05,-1.200000e-05, -2.500000e-05,
            4.400000e-05,  9.800000e-04,  3.640000e-04,  2.030000e-04,-1.830000e-04,  2.400000e-05,
            -2.175000e-03, -3.588000e-03, -1.376000e-03, -8.250000e-04,-7.890000e-04, -3.520000e-04])

for mock_num in np.arange(0,1000):
    # mock_num = 0
    l=l+1             
    
    print('Progress {}%'.format(round(l/len(np.arange(0,1000))*100, 5)))
        
    # Fitting wit curve_fit()    
    mock=read_acf_mocks(mock_num)
        
    try:
        popt, pcov = curve_fit(f=template_generator,
                               xdata=theta_array,
                               ydata=mock,
                               p0=p0,
                               sigma=cov_mat)
    except:
        ValueError
    
    chi_2_array=np.append(chi_2_array,chi_2(theta_array=theta_array,
                                            mock_num=mock_num,
                                            alpha=popt[0],
                                            B_array=popt[1:7],
                                            A0_array=popt[7:13],
                                            A1_array=popt[13:19],
                                            A2_array=popt[19:]))
    alpha_array=np.append(alpha_array,popt[0])
    alpha_error_array=np.append(alpha_error_array,np.sqrt(pcov[0,0]))
        
np.savetxt(os.path.join(path, 'alpha_error.txt'), np.array(np.vstack([alpha_array, alpha_error_array])))
np.savetxt(os.path.join(path, 'chi2.txt'), chi_2_array)
     
tfinal =  datetime.datetime.now()
print(tfinal - tinicial)

"""Analysis of the results"""

plt.hist(alpha_array, bins=16)
plt.xlabel(r'$\alpha$',fontsize=16)
plt.axvline(np.mean(alpha_array),color='r',linestyle='--',label=r'$\langle\alpha\rangle$={}'.format(round(np.mean(alpha_array),5)))
plt.legend()


np.std(alpha_array)/np.mean(alpha_array)*100
np.mean(chi_2_array)

x = np.arange(200, 360, .5)
plt.plot(x, stats.chi2.pdf(x, df=275), color='r', lw=2,label='$\chi^2$(275 dof)')
freq,bine,_=plt.hist(chi_2_array,bins=20,density=True)
plt.xlabel(r'$\chi^2$',fontsize=16)
plt.legend()

