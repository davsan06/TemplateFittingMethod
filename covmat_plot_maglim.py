#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 11:34:56 2020

@author: davsan06
"""

import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

path = '/scratch/davsan06/maglim_BAO/test_binang_1.0_10.0_nbins_50'

cov = np.loadtxt(os.path.join(path, 'covmat_maglim.dat'))

###############################################################################
#######                         Covariance Matrix                     #########
###############################################################################       

ticks_lab= ['z-bin {}'.format(i) for i in np.arange(1,7,1)]
ticks_pos= np.linspace(0, np.shape(cov)[0], 6)    

plt.figure(figsize=(6,6))
sns.heatmap(cov/np.max(cov), square=True, cmap=sns.color_palette("RdBu_r", 1000), cbar_kws={"shrink": .68})
plt.title('MagLim covariance: 6 z-bin x 50 angular bins \n', fontsize=16)
plt.xticks(ticks= ticks_pos,labels=ticks_lab, rotation=45)
plt.yticks(ticks= ticks_pos,labels=ticks_lab, rotation=45)
# plt.savefig(os.path.join(path, 'covmat_maglim.png'),
#             format='png',
#             dpi=150,
#             bbox_inches='tight')


###############################################################################
#######                         Correlation Matrix                     ########
############################################################################### 

corr = np.zeros((np.shape(cov)))

for i in np.arange(np.shape(cov)[0]):
    for j in np.arange(np.shape(cov)[1]):
        
        corr[i,j] = cov[i,j]/(np.sqrt(cov[i,i])*np.sqrt(cov[j,j]))
        
ticks_lab= ['z-bin {}'.format(i) for i in np.arange(1,7,1)]
ticks_pos= np.linspace(0, np.shape(cov)[0], 6)    

plt.figure(figsize=(6,6))
sns.heatmap(corr, square=True, cmap=sns.color_palette("RdBu_r", 1000), cbar_kws={"shrink": .68})
plt.title('MagLim correlation: 6 z-bin x 50 angular bins \n', fontsize=16)
plt.xticks(ticks= ticks_pos,labels=ticks_lab, rotation=45)
plt.yticks(ticks= ticks_pos,labels=ticks_lab, rotation=45)
# plt.savefig(os.path.join(path, 'corrmat_maglim.png'),
#             format='png',
#             dpi=150,
#             bbox_inches='tight')