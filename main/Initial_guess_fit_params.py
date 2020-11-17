#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 13:00:42 2020

@author: davsan06
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 13:01:54 2020

@author: davsan06
"""

# import camb
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


path='/scratch/davsan06/Doctorado/maglim_BAO'
path_acfs_mocks = os.path.join(path, 'test_binang_1.0_10.0_nbins_50')
path_model = os.path.join(path, 'acf_model')

# Angular binning
theta_array = np.loadtxt(os.path.join(path_acfs_mocks, 'theta_acf_bin_0'))[:,0]
    
###############################################################################
#######                     TEMPLATE METHOD                           #########        
###############################################################################
def template_generator(theta_array,alpha,B,A0,A1,A2):
    # alpha=1.01
    # B = 3.0
    # A0 = 0.0
    # A1=-5e-4
    # A2=0.0
    # z_bin=3
        
    theta_model, acf_model=np.loadtxt(os.path.join(path_model, 'theta_acf_maglim_zbin_{}.txt'.format(int(z_bin))))[0], np.loadtxt(os.path.join(path_model, 'theta_acf_maglim_zbin_{}.txt'.format(int(z_bin))))[1]
      
    #  Sin Extrpolar!
    acf_model = interp1d(theta_model, acf_model)   
    
    template = B*acf_model(alpha*theta_array) + A0*np.ones_like(theta_array) + np.divide(A1, theta_array) + np.divide(A2, theta_array**2)

    return(template)
    
for z_bin in np.arange(1,7):
    #z_bin=1
    
    print('z-bin={}'.format(z_bin))
    
    mean = np.loadtxt(os.path.join(path_acfs_mocks,'theta_acf_bin_{}'.format(z_bin-1)))[:,1]

    popt, pcov = curve_fit(template_generator,theta_array,mean)

    temp = template_generator(theta_array,
                              alpha=popt[0],
                              B=popt[1],
                              A0=popt[2],
                              A1=popt[3],
                              A2=popt[4])
    
    print([x for x in popt])
    print([np.sqrt(x) for x in np.diag(pcov)])
    
    chi2=np.sum((mean-temp)**2/temp)
    print(chi2)
    
    plt.plot(theta_array,theta_array**2*temp,label='Template')
    plt.plot(theta_array,theta_array**2*mean,label='Mean mocks')
    plt.legend()
    plt.show()
     
        
        