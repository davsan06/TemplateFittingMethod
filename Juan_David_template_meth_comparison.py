#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:26:00 2020

@author: davsan06
"""

import os
import numpy as np
import matplotlib.pyplot as plt

path_david = '/scratch/davsan06/maglim_BAO/acf_model'
path_juan = '/afs/ciemat.es/user/d/davsan06/Documentos/Datos/omega_theta_th_mocks'
path_to_save = os.path.join(path_david, 'juan_david_template_comparison')

""" Comparison Figures """
for z_bin in np.arange(1,7):
    #z_bin=np.arange(1,7)[0]
    
    acf_david = np.loadtxt(os.path.join(path_david,'theta_acf_maglim_zbin_{}.txt'.format(z_bin)))
    acf_juan = np.loadtxt(os.path.join(path_juan, 'omega_theta_th_bin{}.txt'.format(z_bin-1)))    

    plt.semilogx(acf_juan[0]*180/np.pi, (acf_juan[0]*180/np.pi)**2*acf_juan[1], label='Juan')
    plt.semilogx(acf_david[0], acf_david[0]**2*acf_david[1], label='David')
    plt.title('ACF template fitting method, MagLim z-bin {}'.format(z_bin), fontsize=18)
    plt.xlabel('$log_{10}(\Theta)$ (deg.)', fontsize=16)
    plt.ylabel('$\Theta^2 \cdot w(\Theta)$', fontsize=16)
    plt.legend()
    plt.savefig(os.path.join(path_to_save, 'template_comparison_zbin_{}.png'.format(z_bin)),
                dpi=150,
                format='png',
                bbox_inches='tight')
    plt.show()
    