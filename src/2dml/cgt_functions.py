##
## cgt_functions.py
## Takes functions from CGT_share and places them here...
##
## Created Jul 2, 2018

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
#from urlparse import urljoin
#from bs4 import BeautifulSoup
import pickle
import os.path
import seaborn as sns
from ase.spacegroup import Spacegroup
# import pubchempy as pcp
# creates: band_alignment.png
from math import floor, ceil
import ase.db
from matplotlib import cm
import math
import operator
from mbtr_functions import *
from pymatgen import Lattice, Structure, Molecule
import re
import pymatgen as mp

def energyplot(spinMatrix_dif, title, cmaplabel,vmin,vmax,range=True):
    """
        constructe 2D energy difference plot using Matrix input
    """
    current_cmap = matplotlib.cm.get_cmap(name=cmaplabel)
    current_cmap.set_bad(color='grey')
    if range == True:
        plt.imshow(spinMatrix_dif, cmap=current_cmap,vmin=vmin,vmax=vmax)
        #plt.imshow(spinMatrix_dif, cmap=my_cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(spinMatrix_dif, cmap=current_cmap)
    Batoms = [' '.join(x) for x in B_atom_pair]
    x = np.arange(spinMatrix_dif.shape[1])
    y = np.arange(spinMatrix_dif.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.title(title)
    plt.grid(False)
    plt.xticks(x, xlabels,rotation='vertical', fontsize=35)
    plt.yticks(y, ylabels,rotation='horizontal', fontsize=35)
    font_size = 35 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)
    return



def X2_gen(df_nna,target_label,is_target=True):
    """ 
        Assigns subset of features from df_nna to X2 
    """       
    limited = True
    if limited == False:
        X2 = df_nna[['num_p', 'num_d', 'num_f', 'atomic_rad', 
                     'atomic_vol', 'covalent_rad','dipole','eaffinity','num_electrons',
                     'atomic_rad_avg','atomic_rad_max_dif', 'atomic_rad_sum_dif',
                         'atomic_rad_std_dif','atomic_rad_std',
                         'atomic_vol_avg', 'atomic_vol_max_dif','atomic_vol_sum_dif',
                         'atomic_vol_std_dif','atomic_vol_std',
                         'covalentrad_avg', 'covalentrad_max_dif', 'covalentrad_sum_dif', 
                         'covalentrad_std_dif','covalentrad_std', 
                         'dipole_avg', 'dipole_max_dif','dipole_sum_dif',
                         'dipole_std_dif','dipole_std',
                         'numelectron_avg', 'numelectron_max_dif', 'numelectron_sum_dif', 
                         'numelectron_std_dif', 'numelectron_std', 
                         'vdwradius_avg', 'vdwradius_max_dif','vdwradius_sum_dif',
                         'vdwradius_std_dif', 'vdwradius_std',
                         'e_negativity_avg',  'e_negativity_max_dif', 'e_negativity_sum_dif', 
                         'e_negativity_std_dif',  'e_negativity_std', 
                         'nvalence_avg','nvalence_max_dif', 'nvalence_sum_dif', 
                         'nvalence_std_dif',  'nvalence_std', 
                         'lastsubshell_avg',
                         'cmpd_skew_p', 'cmpd_skew_d','cmpd_skew_f', 'cmpd_sigma_p', 
                          'atomE_AB', 'frac_f ', 'std_ion', #'cmpd_sigma_d', 'cmpd_sigma_f',
                         'sum_ion', 'mean_ion']]
    else:        
        #ORIGINAL May 5th:  #USED To make Hf calc but not so good for mag prediction jjun 12
        X2 = df_nna[['hardness_var','hardness_mean',
             'atomic_rad_avg','atomic_rad_max_dif',
             'atomic_rad_sum_dif', 'atomic_rad_std_dif','atomic_rad_std',
             'atomic_vol_avg', 'atomic_vol_max_dif','atomic_vol_sum_dif',
             'atomic_vol_std_dif','covalentrad_avg', 'covalentrad_max_dif', 
             'covalentrad_std_dif','dipole_avg', 'dipole_max_dif',
             'dipole_std_dif','dipole_std',
             'numelectron_avg', 'numelectron_sum_dif', 
             'nvalence_avg','nvalence_max_dif', 'nvalence_sum_dif', 
             'nvalence_std_dif',  'nvalence_std', 
             'std_ion', 'sum_ion',
             'Nup_mean','Nup_var']]   ## ADDED Nup_mean Nup_var after doing Hf TEST calculation

        # TOP 18 descriptors only
        #
        #         X2 = df_nna[['hardness_mean', 'Nup_mean', 'sum_ion', 'nvalence_std_dif',
        #        'covalentrad_avg', 'covalentrad_max_dif', 'atomic_vol_avg',
        #        'std_ion', 'dipole_std', 'atomic_rad_std_dif',
        #        'covalentrad_std_dif', 'hardness_var', 'nvalence_std',
        #        'nvalence_max_dif', 'nvalence_sum_dif', 'nvalence_avg',
        #        'dipole_avg', 'dipole_max_dif']]
        
    if is_target == True:
        #target = X2[[u'cohesive']] 
        target = df_nna[[target_label]] 
    else:
        target = None
    # Remove all product info :
    # remove magmom since not so necessary and don't have this info for exo_data
    drop_features = [[u'atomic_rad',u'atomic_vol', u'covalent_rad',
                      u'dipole', u'eaffinity',u'num_electrons',u'lastsubshell_avg']]
    formulas = df_nna['formula']
    return X2, target, formulas



def df_to_X(df_test_counts, target_label, is_target, recalculate):
    """
        add mendeleev data to to initiali dataframe.  
        add Hardness info also (std and mean)
        Add N_spin_up info, add to the chemical space also 
        Remove NaNs as needed
        Create X2 and target data. Target data are all zeros if none_ exist
    """
    df1_test = df_test_counts.copy(deep=True)
    data_name = 'mendeleevdata_test'
    picklefile = data_name + '.p'        
    # recalculate = False

    filepath = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML' + picklefile
    (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron, ionenergies, oxi, vdwradius, 
    en, nvalence, elem_list, weights, lastshell, boiling_point, density, evaporation_heat, fusion_heat, gas_basicity, 
    heat_of_formation, melting_point, thermal_conductivity) = gen_mendel_data(df1_test, recalculate, filepath, data_name)

    #print('nvanelce shape', nvalence.shape)
    #print('cmpd_skew_p', cmpd_skew_p.shape)
    df1_test = build_ABX_mendel(df1_test, atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron, ionenergies, oxi, vdwradius, 
    en, nvalence, elem_list, weights, lastshell, boiling_point, density, evaporation_heat, fusion_heat, gas_basicity, 
    heat_of_formation, melting_point, thermal_conductivity)

    #print('df1.shap befoer hard', df1_test.shape)
    # Add Hardness info here
    hard_mean, hard_var = calc_hardness_stats(df1_test['elem_list'])
    df1_test['hardness_mean'] = hard_mean
    df1_test['hardness_var'] = hard_var

    # Add spin states info 
    Nup_mean, Nup_var = gen_spin_stats(df1_test)
    #print('len(Nup_mean)', len(Nup_mean), 'shape0', df1_test.shape)
    df1_test['Nup_mean'] = Nup_mean
    df1_test['Nup_var'] = Nup_var
    
    #print('df1.shap after nup', df1_test.shape)
    # Drop features that have sone NA entries
    df1_test_nna = df1_test.dropna(axis=1)
    X2, t, formula = X2_gen(df1_test_nna,target_label, is_target)
    return X2, t, formula, df1_test_nna



def gen_cohesive(df_input,df_elements):
    """ calculates cohesive energy from total energy and sum of atomic energies"""
    df = df_input.copy()
    total_elem_energy,elems_list_set,frac_dict_list = get_atom_energy_total(df, df_elements)
    df['total_elem_energy'] = total_elem_energy
    df['elem_frac'] = frac_dict_list
    df['elem_list'] = elems_list_set

    cohesive = df['energy'].values - df['total_elem_energy'].values
    df['cohesive'] = cohesive
    return df




def spinMatrixGen(df_counts,TMlist,B_atom_pair,descriptor):
    """ 
        generates EnergyMAtrix from B list, TM list
        constrain for Te containing, or Se containing atoms etc prior
    """
    df_counts1 = df_counts.copy(deep=True)
    df_counts1 = df_counts1.reset_index()
    spinMatrix = np.ones((len(TMlist),len(B_atom_pair)))*3.141
    #print(TMlist)
    #print(B_atom_pair)
    for i,cmpd in enumerate(df_counts1['formula'][:]):
        #print(cmpd)
        for bth, b in enumerate(B_atom_pair):
            for tmth, tm in enumerate(TMlist):
                #print(b, tm)
                if tm == 'Cr': #special case when Cr is tm. always will find it ipresent...
                    TMtrue = df_counts1.loc[i,'Cr'] == 2.0
                    #print(TMtrue)
                    Btrue = Bexists(i, b, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_spinstate = df_counts1['spin_state'][i]
                        TMB_spinstate = df_counts1[descriptor][i]
                        if pd.isnull(TMB_spinstate):
                            spinMatrix[tmth,bth] = np.nan
                        else:        
                            spinMatrix[tmth,bth] = TMB_spinstate
                else:
                    Btrue = Bexists(i, b, df_counts1)
                    TMtrue = TMexists(i, tm, df_counts1)
                    if Btrue and TMtrue:
                        TMB_spinstate = df_counts1[descriptor][i]
                        if pd.isnull(TMB_spinstate):
                            spinMatrix[tmth,bth] = np.nan
                        else:
                            spinMatrix[tmth,bth] = TMB_spinstate
    spinMatrix_nna = spinMatrix.copy()
    spinMatrix_nna[np.isnan(spinMatrix_nna)] = np.nan #-1.0
    return spinMatrix_nna



def cohesive_matrix_plot(df_spin_Te, df_elements,TMlist, B_atom_pair,title ):
    # Create atom counts:
    atom_label_list_spin, atom_count_list_spin = get_atom_counts(df_spin_Te, df_elements)
    df_spin_counts_Te = df_spin_Te.copy(deep = True)
    for ith, atom_label in enumerate(atom_label_list_spin):
        #print(atom_label)
        df_spin_counts_Te[atom_label] = atom_count_list_spin[ith]
        #atom_count_list

    # print(B_atom_pair)
    descriptor = 'cohesive'
    spinMatrix_Te = spinMatrixGen(df_spin_counts_Te,TMlist,B_atom_pair,descriptor)

    # Create plot
    
    cmaplabel = 'inferno'
    plt.figure(figsize=(8.5,8.5))
    vmin = -8; vmax=0.5;
    energyplot(spinMatrix_Te, title, cmaplabel,vmin,vmax,range=True)
    #plt.colorbar()
    return


def plot_mag(spinMatrix_mag,title,Batoms,TMlist,vmin,vmax):
    """ plots magnetic moments from spinMAtrix """
    plt.figure(figsize=(6,6))

    cmap='Blues'
    current_cmap = matplotlib.cm.get_cmap('inferno')
    current_cmap.set_bad(color='grey')

    # plt.imshow(spinMatrix_dif, cmap='inferno')
    plt.imshow(spinMatrix_mag, cmap=current_cmap,vmin=vmin,vmax=vmax)

    # plt.colorbar(label='magnetic moment / unit cell')
    Batoms = [' '.join(x) for x in B_atom_pair]

    font_size = 20 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)

    x = np.arange(spinMatrix_mag.shape[1])
    y = np.arange(spinMatrix_mag.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.grid(False)
    plt.title(title)
    plt.xticks(x, xlabels,rotation='vertical',fontsize=20)
    plt.yticks(y, ylabels,rotation='horizontal',fontsize=20)
    plt.show()
    
    
def plot_elem(df_counts,elem,label,nbins):
    #
    binwidth = 0.5
    e_other = df_counts[label][df_counts[elem] == 0].values
    binsinfo =np.arange(np.min(e_other),np.max(e_other)+binwidth,binwidth)
    #
    df_counts[label][df_counts[elem] == 0].hist(color='g', bins=binsinfo, alpha=0.6,normed=True)
    df_counts[label][df_counts[elem] >= 1].hist(color='r', bins=binsinfo, alpha=0.6,normed=True)
    #plt.xlabel('cohesive energy',fontsize=30)
    #plt.ylabel('counts',fontsize=30)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.grid(False)
    plt.locator_params(nbins=4,axis='y')
    plt.ylim(0,0.65)
    plt.xlim(-8.5,1.5)
    plt.legend()
    return



def EnergyMatrixGen(df_counts,TMlist,B_atom_pair, label):
    """ 
        generates EnergyMAtrix from B list, TM list
        constrain for Te containing, or Se containing atoms etc prior
        - label -> 'cohesive', 'magmom' etc.
        -1/10: corrected error with Cr atom always being found and index being updated erroneously..
    """
    df_counts1 = df_counts.copy(deep=True)
    df_counts1 = df_counts1.reset_index()
    EnergyMatrix = np.empty((len(TMlist),len(B_atom_pair)))
    print(EnergyMatrix.shape)
    #print(EnergyMatrix)
    for i,cmpd in enumerate(df_counts1['formula'][:]):
        for bth, b in enumerate(B_atom_pair):
            for tmth, tm in enumerate(TMlist):
                if tm == 'Cr': #special case when Cr is tm. always will find it ipresent...
                    TMtrue = df_counts1.loc[i,'Cr'] == 2.0
                    #print(TMtrue)
                    Btrue = Bexists(i, b, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_energy = df_counts1['cohesive'][i]
                        TMB_energy = df_counts1[label][i]
                        #print(tm, b, TMB_energy)
                        EnergyMatrix[tmth,bth] = TMB_energy
                else:
                    #print(i,bth,tmth)
                    Btrue = Bexists(i, b, df_counts1)
                    TMtrue = TMexists(i, tm, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_energy = df_counts1['cohesive'][i]
                        TMB_energy = df_counts1[label][i]
                        EnergyMatrix[tmth,bth] = TMB_energy
    EnergyMatrix_nna = EnergyMatrix.copy()
    EnergyMatrix_nna[np.isnan(EnergyMatrix_nna)] = np.nan
    return EnergyMatrix_nna




def energyscape(df_counts,X,feature_label,TMlist,B_atom_pair,vmin,vmax):
    """  
        creates cohesive energy 2D plot 
    """
    df_counts_Te = df_counts[df_counts[X] ==6].copy()
    df_counts_Te_cp = df_counts_Te.copy()
    #feature_label = 'cohesive'
    EnergyMatrix_nna = EnergyMatrixGen(df_counts_Te_cp,TMlist,B_atom_pair,feature_label)

    plt.figure(figsize=(12,8))
    #cmap='spectral'
    current_cmap = matplotlib.cm.get_cmap('coolwarm')
    current_cmap.set_bad(color='white')
    #vmin = -4
    #vmax = 2.0
    plt.imshow(EnergyMatrix_nna,cmap=current_cmap, interpolation='none',vmin=vmin,vmax=vmax)
    
    #plt.colorbar()
    font_size = 20 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)
    
    Batoms = [' '.join(x) for x in B_atom_pair]
    print(len(Batoms), Batoms)
    print(EnergyMatrix_nna.shape)
    x = np.arange(EnergyMatrix_nna.shape[1])
    y = np.arange(EnergyMatrix_nna.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.grid(False)
    #plt.title(' TEST predition')
    plt.xticks(x, xlabels,rotation='vertical',fontsize=20)
    plt.yticks(y, ylabels,rotation='horizontal',fontsize=20)
    plt.show()
    return EnergyMatrix_nna

def get_mse(y_test,prediction):
    """
        Calculates the Mean Squared Error of test data and predictions
    """
    acc = np.mean((y_test-prediction)**2.0);
    mean_dif = np.mean(np.abs(y_test-prediction))
    return acc, mean_dif



