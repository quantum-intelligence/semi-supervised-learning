#
# df_gen.py
#

# UPDATES:
# 9/7/2018: get_df_optima(df_master) to include Delta_FM_AFM states
# updated 10.5.2018 to work with MAE (yiqi) calculations (i,e. spin orbit coupling only)
# updated get_df_optima to include 'energy'

import os
import pymatgen as mg
import pymatgen as mp

from math import floor, ceil
import itertools
import re
import numpy as np
import matplotlib.pyplot as plt
import ase.db
from mendeleev import element
from sklearn.decomposition import PCA
import scipy as sp
from sklearn.preprocessing import PolynomialFeatures
from ase.db.plot import dct2plot
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.model_selection import validation_curve
from sklearn.linear_model import Lasso
import random
#mpr = MPRester("0Eq4tbAMNRe37DPs")
from pymatgen.ext.matproj import MPRester
m = MPRester("RK5GrTk1anSOmgAU")

from scipy.stats import skew
import shutil

from fmmlcalc_b import *
from alloy_functions import *



def master_mag_df():
    """ gather magnetic moment info for all spin configurations"""
    edir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/energy_results'

    #     efile_spin_Te = edir + '/magmom_results_spin_Te.csv'
    #     df_spin_Te = pd.read_csv (efile_spin_Te, delimiter=',', usecols=[1,2])
    #     df_spin_Te = df_spin_Te.rename(columns={'mag_mom':'magmom_spin'})
    #     df_spin_Te = df_spin_Te.rename(columns={'formula':'formula_spin'})
    #     #
    #     efile_spin_Se = edir + '/magmom_results_spin_Se.csv'
    #     df_spin_Se = pd.read_csv (efile_spin_Se, delimiter=',', usecols=[1,2])
    #     df_spin_Se = df_spin_Se.rename(columns={'mag_mom':'magmom_spin'})
    #     df_spin_Se = df_spin_Se.rename(columns={'formula':'formula_spin'})
    #     #
    #     efile_spin_S = edir + '/magmom_results_spin_S.csv'
    #     df_spin_S = pd.read_csv (efile_spin_S, delimiter=',', usecols=[1,2])
    #     df_spin_S = df_spin_S.rename(columns={'mag_mom':'magmom_spin'})
    #     df_spin_S = df_spin_S.rename(columns={'formula':'formula_spin'})

    efile_spin_so_S = edir + '/magmom_results_spin_so_S.csv'
    df_spin_so_S = pd.read_csv (efile_spin_so_S, delimiter=',', usecols=[1,2])
    df_spin_so_S = df_spin_so_S.rename(columns={'mag_mom':'magmom_spin_so'})
    df_spin_so_S = df_spin_so_S.rename(columns={'formula':'formula_spin_so'})
    #
    efile_spin_so_Se = edir + '/magmom_results_spin_so_Se.csv'
    df_spin_so_Se = pd.read_csv (efile_spin_so_Se, delimiter=',', usecols=[1,2])
    df_spin_so_Se = df_spin_so_Se.rename(columns={'mag_mom':'magmom_spin_so'})
    df_spin_so_Se = df_spin_so_Se.rename(columns={'formula':'formula_spin_so'})
    #
    efile_spin_so_Te = edir + '/magmom_results_spin_so_Te.csv'
    df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
    df_spin_so_Te = df_spin_so_Te.rename(columns={'mag_mom':'magmom_spin_so'})
    df_spin_so_Te = df_spin_so_Te.rename(columns={'formula':'formula_spin_so'})

    efile_afm_Te = edir + '/magmom_results_afm_Te.csv'
    df_afm_Te = pd.read_csv (efile_afm_Te, delimiter=',', usecols=[1,2])
    df_afm_Te = df_afm_Te.rename(columns={'mag_mom':'magmom_afm'})
    df_afm_Te = df_afm_Te.rename(columns={'formula':'formula_afm'})
    #
    efile_afm_Se = edir + '/magmom_results_afm_Se.csv'
    df_afm_Se = pd.read_csv (efile_afm_Se, delimiter=',', usecols=[1,2])
    df_afm_Se = df_afm_Se.rename(columns={'mag_mom':'magmom_afm'})
    df_afm_Se = df_afm_Se.rename(columns={'formula':'formula_afm'})
    #
    efile_afm_S = edir + '/magmom_results_afm_S.csv'
    df_afm_S = pd.read_csv (efile_afm_S, delimiter=',', usecols=[1,2])
    df_afm_S = df_afm_S.rename(columns={'mag_mom':'magmom_afm'})
    df_afm_S = df_afm_S.rename(columns={'formula':'formula_afm'})

    # df_spin = pd.concat((df_spin_Te,df_spin_Se,df_spin_S))
    # print(df_spin_so_Te['formula_spin_so'][:10])
    # print(df_afm_Te['formula_afm'][:10])
    df_spin_so = pd.concat((df_spin_so_Te,df_spin_so_Se,df_spin_so_S))
    df_spin_so = df_spin_so.sort_values('formula_spin_so')
    df_afm = pd.concat((df_afm_Te, df_afm_Se, df_afm_S))
    df_afm = df_afm.sort_values('formula_afm')

    # df_mag_tot = pd.concat((df_spin, df_spin_so, df_afm),axis=1)
    df_mag_tot = pd.concat((df_spin_so, df_afm),axis=1)
    #formula = df_mag_tot['formula'].values
    formula = df_spin_so['formula_spin_so'].values

    df_mag_tot['formula'] = formula
    #df_mag_tot.rename(columns={'elem_list':'elem_list_mag','frac_list':'frac_list_mag'})
    df_mag_tot = df_mag_tot.sort_values('formula')
    df_mag_tot = df_mag_tot.reset_index()
    df_mag_tot = df_mag_tot.drop(columns=['index'])
    return df_mag_tot



def add_atom_counts_mag(main_dir, df_mag,mag_label, recalculate):
    """
    adds atom counts to dataframe
    saved to df_gen.py
    """
    # Calculate df_elements_mag
    #recalculate = False
    df_elements_mag = get_unique_elem_info(main_dir, df_mag, recalculate=recalculate)
    #
    total_elem_energy, elems_list_set, frac_dict_list = get_atom_energy_total(df_mag, df_elements_mag)
    df_mag['elem_frac'] = frac_dict_list
    df_mag['elem_list'] = elems_list_set
    #
    atom_label_list_mag, atom_count_list_mag = get_atom_counts(df_mag, df_elements_mag, 'elem_list')
    df_mag_counts = df_mag.copy(deep = True)
    for ith, atom_label in enumerate(atom_label_list_mag):
        df_mag_counts[atom_label] = atom_count_list_mag[ith]

    mag_mom = df_mag_counts[mag_label].values
    if mag_label == 'mag_mom':
        mag_mom = np.abs(mag_mom)
    df_mag_counts.loc[:,mag_label] = mag_mom
    return df_mag_counts



def get_df_optima(df_master):
    """
        Create df with min E configurations
        saved to df_gen.py
        - modified 12.5.2018 to no longer accept values wtihout spin orbit coupling
    """
    #spinlabel = ['spin','spin_so','afm']
    spinlabel = ['spin_so','afm']
    minE = []
    optMag = []
    optDft = []
    spinlabels = []
    spindex = []
    print(df_master.columns)
    delta_FM_AFM = []
    for c in df_master[:].iterrows():
        rownum = c[0]
        #energies = (c[1][['cohesive_spin','cohesive_spin_so','cohesive_afm']])
        energies = (c[1][['cohesive_spin_so','cohesive_afm']])
        dft_energy = c[1][['energy_spin_so','energy_afm']]
        fm_energies = c[1]['cohesive_spin_so']
        afm_energies = c[1]['cohesive_afm']
        D_fm_afm = fm_energies - afm_energies
        #magmoms = (c[1][['magmom_spin','magmom_spin_so','magmom_afm']])
        magmoms = (c[1][['magmom_spin_so','magmom_afm']])
        magmoms = np.abs(magmoms)
        mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
        optimum_mag = magmoms[mindex_energy]
        opti_dft_energy = dft_energy[mindex_energy]
        optMag.append(optimum_mag)
        optDft.append(opti_dft_energy)
        minE.append(np.min(energies))
        spindex.append(mindex_energy)
        spinlabels.append(spinlabel[mindex_energy])
        delta_FM_AFM.append(D_fm_afm)
    df_optima = pd.DataFrame()
    df_optima['formula'] = df_master['formula'].values
    df_optima['cohesive'] = minE #added 12.23.2018  #cant rename 'energy'
    df_optima['energy'] = optDft #added 12.23.2018  #cant rename 'energy'
    df_optima['total_elem_energy'] = df_master['total_elem_energy'].values
    df_optima['minE'] = minE
    df_optima['optMag'] = optMag
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    df_optima['delta_FM_AFM'] = delta_FM_AFM
    return df_optima

import math

#####


def get_df_optima_tmx(df_master):
    """
        Create df with min E configurations
        saved to df_gen.py
        - modified 12.5.2018 to no longer accept values wtihout spin orbit coupling
        - modified 5.7.2019 for TMX compounds
    """
    spinlabel = ['spin_so','afm_so']
    minE = []
    optMag = []
    optDft = []
    spinlabels = []
    spindex = []
    delta_FM_AFM = []
    for c in df_master[:].iterrows():
        rownum = c[0]
        #energies = (c[1][['spin_so_energy','afm_so_energy']])
        #print('energies',energies)
        if 'spin_so_energy' in df_master.columns:
            #energies = (c[1][['spin_so_energy','afm_so_energy']])  # modified mar 5 20202
            energies = (c[1][['cohesive_spin_so','cohesive_afm_so']]) # modified mar 5 20202
            ##dft_energy = c[1][['spin_so_energy','afm_so_energy']]
            dft_energy = c[1][['spin_so_energy','afm_so_energy']]
            fm_energies = c[1]['spin_so_energy']
            afm_energies = c[1]['afm_so_energy']
        else:
            #print(c[1])
            #print(c[1].loc['energy_spin_so']) #,'frac_f']])
            #energies = [c[1].loc['energy_spin_so'], c[1].loc['energy_afm']] # modified mar 5 20202
            energies = [c[1].loc['cohesive_spin_so'],c[1].loc['cohesive_afm_so']] # modified mar 5 20202
            #dft_energy = c[1][['energy_spin_so','energy_afm_so']]
            dft_energy = [c[1].loc['energy_spin_so'], c[1].loc['energy_afm']] #energy_afm_so renamed energy_afm
            fm_energies = c[1]['energy_spin_so']
            afm_energies = c[1]['energy_afm'] #renamed energy_afm_so --> energy_afm
        D_fm_afm = fm_energies - afm_energies
        magmoms = (c[1][['spin_so_tot_mu','afm_so_tot_mu']])
        #magmoms = np.abs(magmoms)
        if pd.isnull(energies).any():
            #print('find NAN')
            optimum_mag = np.nan
            opti_dft_energy = np.nan
            mindex_energy = np.nan  # added 1.3.2020
            spindex.append(mindex_energy)# added 1.3.2020
            spinlabels.append(np.nan)# added 1.3.2020
        else:
            #print('energies',energies, type(energies))
            #print('min',np.min(energies))
            #mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
            mindex_energy = np.where(energies == np.min(energies))[0][0]
            #print('mindex_energy',mindex_energy)
            optimum_mag = magmoms[mindex_energy]
            opti_dft_energy = dft_energy[mindex_energy]
            spindex.append(mindex_energy)# added 1.3.2020
            spinlabels.append(spinlabel[mindex_energy])# added 1.3.2020
        optMag.append(optimum_mag)
        optDft.append(opti_dft_energy)
        minE.append(np.min(energies))
        #spindex.append(mindex_energy) #moved to above if statementt 1.3.2020
        #spinlabels.append(spinlabel[mindex_energy])
        delta_FM_AFM.append(D_fm_afm)
    #df_optima = pd.DataFrame()
    df_optima = df_master.copy(deep=True) #modified on 3.3.2020
    df_optima['formula'] = df_master['formula'].values
    df_optima['cohesive'] = minE #added 12.23.2018  #cant rename 'energy'
    df_optima['energy'] = optDft #added 12.23.2018  #cant rename 'energy'
    #df_optima['total_elem_energy'] = df_master['total_elem_energy'].values
    df_optima['minE'] = minE
    df_optima['optMag'] = optMag
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    df_optima['delta_FM_AFM'] = delta_FM_AFM
    return df_optima
#copied from jupyter notebook 2.19.2020
######

# #commented out 2.19.2020
#
# def get_df_optima_tmx(df_master):
#     """
#         Create df with min E configurations
#         saved to df_gen.py
#         - modified 12.5.2018 to no longer accept values wtihout spin orbit coupling
#         - modified 5.7.2019 for TMX compounds
#     """
#     spinlabel = ['spin_so','afm_so']
#     minE = []
#     optMag = []
#     optDft = []
#     spinlabels = []
#     spindex = []
#     delta_FM_AFM = []
#     for c in df_master[:].iterrows():
#         rownum = c[0]
#         energies = (c[1][['spin_so_energy','afm_so_energy']])
#         #print('energies',energies)
#         dft_energy = c[1][['spin_so_energy','afm_so_energy']]
#         fm_energies = c[1]['spin_so_energy']
#         afm_energies = c[1]['afm_so_energy']
#         D_fm_afm = fm_energies - afm_energies
#         magmoms = (c[1][['spin_so_tot_mu','afm_so_tot_mu']])
#         #magmoms = np.abs(magmoms)
#         if pd.isnull(energies).any():
#             #print('find NAN')
#             optimum_mag = np.nan
#             opti_dft_energy = np.nan
#             mindex_energy = np.nan  # added 1.3.2020
#             spindex.append(mindex_energy)# added 1.3.2020
#             spinlabels.append(np.nan)# added 1.3.2020
#         else:
#             # mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
#             # modified on 2.17.2020 - Series dont not work with np.argwhere()
#             mindex_energy = np.where(energies == np.min(energies))[0][0]
#             optimum_mag = magmoms[mindex_energy]
#             opti_dft_energy = dft_energy[mindex_energy]
#             spindex.append(mindex_energy)# added 1.3.2020
#             spinlabels.append(spinlabel[mindex_energy])# added 1.3.2020
#         optMag.append(optimum_mag)
#         optDft.append(opti_dft_energy)
#         minE.append(np.min(energies))
#         #spindex.append(mindex_energy) #moved to above if statementt 1.3.2020
#         #spinlabels.append(spinlabel[mindex_energy])
#         delta_FM_AFM.append(D_fm_afm)
#     df_optima = pd.DataFrame()
#     df_optima['formula'] = df_master['formula'].values
#     df_optima['cohesive'] = minE #added 12.23.2018  #cant rename 'energy'
#     df_optima['energy'] = optDft #added 12.23.2018  #cant rename 'energy'
#     #df_optima['total_elem_energy'] = df_master['total_elem_energy'].values
#     df_optima['minE'] = minE
#     df_optima['optMag'] = optMag
#     df_optima['spinlabels'] = spinlabels
#     df_optima['spindex'] = spindex
#     df_optima['delta_FM_AFM'] = delta_FM_AFM
#     return df_optima
#


def get_df_optima_tmx_noso(df_master):
    """
        Create df with min E configurations
        saved to df_gen.py
        - modified 12.5.2018 to no longer accept values wtihout spin orbit coupling
        - modified 5.7.2019 for TMX compounds
        - created 6.29.2019 to consider non-spin orbit coupling calculations
        - updated 1.3.2019 to better account for missing values
    """
    spinlabel = ['spin','afm']
    minE = []
    optMag = []
    optDft = []
    spinlabels = []
    spindex = []
    delta_FM_AFM = []
    for c in df_master[:].iterrows():
        rownum = c[0]
        energies = (c[1][['spin_energy','afm_energy']])
        #print('energies',energies)
        dft_energy = c[1][['spin_energy','afm_energy']]
        fm_energies = c[1]['spin_energy']
        afm_energies = c[1]['afm_energy']
        D_fm_afm = fm_energies - afm_energies
        magmoms = (c[1][['spin_tot_mu','afm_tot_mu']])
        #magmoms = np.abs(magmoms)
        if pd.isnull(energies).any():
            #print('find NAN')
            optimum_mag = np.nan
            opti_dft_energy = np.nan
            mindex_energy = np.nan
            spindex.append(mindex_energy)
            spinlabels.append(np.nan)
        else:
            #print(energies)
            #mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
            # modified 2.17.2020 argwhere doesn't work with Series
            mindex_energy = np.where(energies == np.min(energies))[0][0]
            optimum_mag = magmoms[mindex_energy]
            opti_dft_energy = dft_energy[mindex_energy]
            spindex.append(mindex_energy)
            spinlabels.append(spinlabel[mindex_energy])
        optMag.append(optimum_mag)
        optDft.append(opti_dft_energy)
        minE.append(np.min(energies))
        #spindex.append(mindex_energy)
        #spinlabels.append(spinlabel[mindex_energy])
        delta_FM_AFM.append(D_fm_afm)
    df_optima = df_master.copy(deep=True) #modified on 3.3.2020
    #df_optima = pd.DataFrame()
    df_optima['formula'] = df_master['formula'].values
    df_optima['cohesive'] = minE #added 12.23.2018  #cant rename 'energy'
    df_optima['energy'] = optDft #added 12.23.2018  #cant rename 'energy'
    #df_optima['total_elem_energy'] = df_master['total_elem_energy'].values
    df_optima['minE'] = minE
    df_optima['optMag'] = optMag
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    df_optima['delta_FM_AFM'] = delta_FM_AFM
    return df_optima



## COpied from notebook 12.30.201
#
# def master_energy_df(edir, main_dir, recalculate):
#     """
#        gathers all energy data
#        returns:
#            edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
#                  df_afm_so_Te, df_afm_so_Se, df_afm_so_S]
#        ensure edf is sorted by 'formula'
#     """
#     # recalculate = False
#
#     #     efile_spin_Te = edir + '/energy_results_spin_Te.csv'
#     #     efile_nospin_Te = edir + '/energy_results_nospin_Te.csv'
#     efile_spin_so_Te = edir + '/energy_results_spin_so_Te.csv'
#     efile_afm_so_Te = edir + '/energy_results_afm_Te.csv'
#     #     df_nospin_Te = pd.read_csv(efile_nospin_Te, delimiter=',', usecols=[1,2])
#     #     df_spin_Te = pd.read_csv(efile_spin_Te, delimiter=',', usecols=[1,2])
#     df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
#     df_afm_so_Te = pd.read_csv(efile_afm_so_Te, delimiter=',', usecols=[1,2])
#     #print(df_nospin_Te.shape, df_spin_so_Te.shape, df_afm_so_Te.shape )
#     # Calculate cohesive energies - Te
#     df_elements = get_unique_elem_info(main_dir, df_spin_so_Te, recalculate=recalculate)
#     #     df_nospin_Te = gen_cohesive(df_nospin_Te, df_elements)
#     #     df_spin_Te = gen_cohesive(df_spin_Te, df_elements)
#     df_spin_so_Te = gen_cohesive(df_spin_so_Te, df_elements)
#     df_afm_so_Te = gen_cohesive(df_afm_so_Te, df_elements)
#
#     #     efile_spin_Se = edir + '/energy_results_spin_Se.csv'
#     #     efile_nospin_Se = edir + '/energy_results_nospin_Se.csv'
#     efile_afm_so_Se = edir + '/energy_results_afm_Se.csv'
#     efile_spin_so_Se = edir + '/energy_results_spin_so_Se.csv'
#     #     df_spin_Se = pd.read_csv(efile_spin_Se, delimiter=',', usecols=[1,2])
#     #     df_nospin_Se = pd.read_csv(efile_nospin_Se, delimiter=',', usecols=[1,2])
#     df_afm_so_Se = pd.read_csv(efile_afm_so_Se, delimiter=',', usecols=[1,2])
#     df_spin_so_Se = pd.read_csv(efile_spin_so_Se, delimiter=',', usecols=[1,2])
#     #print(df_spin_Se.shape, df_nospin_Se.shape, df_afm_so_Se.shape, df_spin_so_Se.shape )
#     # Calculate cohesive energies - Se
#     df_elements = get_unique_elem_info(main_dir, df_spin_so_Se, recalculate=recalculate)
#     #     df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
#     #     df_spin_Se = gen_cohesive(df_spin_Se, df_elements)
#     df_spin_so_Se = gen_cohesive(df_spin_so_Se, df_elements)
#     df_afm_so_Se = gen_cohesive(df_afm_so_Se, df_elements)
#
#     #     efile_spin_S = edir + '/energy_results_spin_S.csv'
#     #     efile_nospin_S = edir + '/energy_results_nospin_S.csv'
#     efile_afm_so_S = edir + '/energy_results_afm_S.csv'
#     efile_spin_so_S = edir + '/energy_results_spin_so_S.csv'
#     #     df_spin_S = pd.read_csv(efile_spin_S, delimiter=',', usecols=[1,2])
#     #     df_nospin_S = pd.read_csv(efile_nospin_S, delimiter=',', usecols=[1,2])
#     df_afm_so_S = pd.read_csv(efile_afm_so_S, delimiter=',', usecols=[1,2])
#     df_spin_so_S = pd.read_csv(efile_spin_so_S, delimiter=',', usecols=[1,2])
#     #print(df_spin_S.shape,df_afm_so_S.shape,df_spin_so_S.shape)
#     # Calculate cohesive energies - S
#     df_elements = get_unique_elem_info(main_dir, df_spin_so_S, recalculate=recalculate)
#     #df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
#     #     df_spin_S = gen_cohesive(df_spin_S, df_elements)
#     df_spin_so_S = gen_cohesive(df_spin_so_S, df_elements)
#     df_afm_so_S = gen_cohesive(df_afm_so_S, df_elements)
#
#     # edf_spin = pd.DataFrame()
#     #     edf_spin = pd.concat((df_spin_Te, df_spin_Se, df_spin_S))
#     #     edf_spin = edf_spin.rename(columns={'energy':'energy_spin'})
#     #     edf_spin = edf_spin.rename(columns={'cohesive':'cohesive_spin'})
#     # edf_spin = edf_spin.rename(columns={'formula':'formula_spin'})
#
#     #CONCATENATE different spin ocnfigurations & relabel columns
#     edf_spin_so = pd.concat((df_spin_so_Te, df_spin_so_Se, df_spin_so_S))
#     edf_spin_so = edf_spin_so.rename(columns={'energy':'energy_spin_so'})
#     edf_spin_so = edf_spin_so.rename(columns={'cohesive':'cohesive_spin_so'})
#     #     edf_spin_so = edf_spin_so.rename(columns={'formula':'formula_spin_so'})
#     edf_spin_so = edf_spin_so.sort_values('formula')
#     edf_spin_so = edf_spin_so.rename(columns={'elem_list':'elem_list_spin_so'}) # Added Dec 13, 2018
#     edf_spin_so = edf_spin_so.rename(columns={'frac_list':'frac_list_spin_so'}) # Added Dec 13, 2018
#     edf_spin_so = edf_spin_so.rename(columns={'elem_frac':'elem_frac_edf'}) # Added Dec 13, 2018
#
#     edf_afm_so = pd.concat((df_afm_so_Te, df_afm_so_Se, df_afm_so_S))
#     edf_afm_so = edf_afm_so.rename(columns={'energy':'energy_afm'})
#     edf_afm_so = edf_afm_so.rename(columns={'cohesive':'cohesive_afm'})
#     edf_afm_so = edf_afm_so.rename(columns={'formula':'formula_afm'})
#     edf_afm_so = edf_afm_so.sort_values('formula_afm')
#     print('change names2',edf_afm_so.columns)
#     edf_afm_so = edf_afm_so.rename(columns={'elem_list':'elem_list_afm'}) # Added Dec 13, 2018
#     edf_afm_so = edf_afm_so.rename(columns={'frac_list':'frac_list_afm'}) # Added Dec 13, 2018
#     edf_afm_so = edf_afm_so.rename(columns={'elem_frac':'elem_frac_edf_afm'}) # Added Dec 13, 2018
#     edf_afm_so = edf_afm_so.rename(columns={'total_elem_energy':'total_elem_energy_afm'})
#
#     #     edf = pd.concat((edf_spin, edf_spin_so, edf_afm_so),axis=1)
#     edf = pd.concat((edf_spin_so, edf_afm_so), axis=1)
#     edf = edf.sort_values('formula')
#     edf = edf.drop(columns=['formula_afm'])#,'total_elem_energy'])
#     #edf = edf.rename(columns={'cohesive_spin_so':'cohesive'}) # Added Dec 13, 2018
#     return edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
#                  df_afm_so_Te, df_afm_so_Se, df_afm_so_S]


# version below had elem_fram_edf error
#
def master_energy_df(edir, main_dir, recalculate):
    """
       gathers all energy data
       returns:
           edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
                 df_afm_so_Te, df_afm_so_Se, df_afm_so_S]
       ensure edf is sorted by 'formula'
    """
    # recalculate = False

    #     efile_spin_Te = edir + '/energy_results_spin_Te.csv'
    #     efile_nospin_Te = edir + '/energy_results_nospin_Te.csv'
    efile_spin_so_Te = edir + '/energy_results_spin_so_Te.csv'
    efile_afm_so_Te = edir + '/energy_results_afm_Te.csv'
    #     df_nospin_Te = pd.read_csv(efile_nospin_Te, delimiter=',', usecols=[1,2])
    #     df_spin_Te = pd.read_csv(efile_spin_Te, delimiter=',', usecols=[1,2])
    df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
    df_afm_so_Te = pd.read_csv(efile_afm_so_Te, delimiter=',', usecols=[1,2])
    #print(df_nospin_Te.shape, df_spin_so_Te.shape, df_afm_so_Te.shape )
    # Calculate cohesive energies - Te
    df_elements = get_unique_elem_info(main_dir, df_spin_so_Te, recalculate)
    #     df_nospin_Te = gen_cohesive(df_nospin_Te, df_elements)
    #     df_spin_Te = gen_cohesive(df_spin_Te, df_elements)
    df_spin_so_Te = gen_cohesive(df_spin_so_Te, df_elements)
    df_afm_so_Te = gen_cohesive(df_afm_so_Te, df_elements)

    #     efile_spin_Se = edir + '/energy_results_spin_Se.csv'
    #     efile_nospin_Se = edir + '/energy_results_nospin_Se.csv'
    efile_afm_so_Se = edir + '/energy_results_afm_Se.csv'
    efile_spin_so_Se = edir + '/energy_results_spin_so_Se.csv'
    #     df_spin_Se = pd.read_csv(efile_spin_Se, delimiter=',', usecols=[1,2])
    #     df_nospin_Se = pd.read_csv(efile_nospin_Se, delimiter=',', usecols=[1,2])
    df_afm_so_Se = pd.read_csv(efile_afm_so_Se, delimiter=',', usecols=[1,2])
    df_spin_so_Se = pd.read_csv(efile_spin_so_Se, delimiter=',', usecols=[1,2])
    #print(df_spin_Se.shape, df_nospin_Se.shape, df_afm_so_Se.shape, df_spin_so_Se.shape )
    # Calculate cohesive energies - Se
    df_elements = get_unique_elem_info(main_dir, df_spin_so_Se, recalculate=recalculate)
    #     df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
    #     df_spin_Se = gen_cohesive(df_spin_Se, df_elements)
    df_spin_so_Se = gen_cohesive(df_spin_so_Se, df_elements)
    df_afm_so_Se = gen_cohesive(df_afm_so_Se, df_elements)

    #     efile_spin_S = edir + '/energy_results_spin_S.csv'
    #     efile_nospin_S = edir + '/energy_results_nospin_S.csv'
    efile_afm_so_S = edir + '/energy_results_afm_S.csv'
    efile_spin_so_S = edir + '/energy_results_spin_so_S.csv'
    #     df_spin_S = pd.read_csv(efile_spin_S, delimiter=',', usecols=[1,2])
    #     df_nospin_S = pd.read_csv(efile_nospin_S, delimiter=',', usecols=[1,2])
    df_afm_so_S = pd.read_csv(efile_afm_so_S, delimiter=',', usecols=[1,2])
    df_spin_so_S = pd.read_csv(efile_spin_so_S, delimiter=',', usecols=[1,2])
    #print(df_spin_S.shape,df_afm_so_S.shape,df_spin_so_S.shape)
    # Calculate cohesive energies - S
    df_elements = get_unique_elem_info(main_dir, df_spin_so_S, recalculate=recalculate)
    #df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
    #     df_spin_S = gen_cohesive(df_spin_S, df_elements)
    df_spin_so_S = gen_cohesive(df_spin_so_S, df_elements)
    df_afm_so_S = gen_cohesive(df_afm_so_S, df_elements)

    # edf_spin = pd.DataFrame()
    #     edf_spin = pd.concat((df_spin_Te, df_spin_Se, df_spin_S))
    #     edf_spin = edf_spin.rename(columns={'energy':'energy_spin'})
    #     edf_spin = edf_spin.rename(columns={'cohesive':'cohesive_spin'})
    # edf_spin = edf_spin.rename(columns={'formula':'formula_spin'})

    #CONCATENATE different spin ocnfigurations & relabel columns
    edf_spin_so = pd.concat((df_spin_so_Te, df_spin_so_Se, df_spin_so_S))
    edf_spin_so = edf_spin_so.rename(columns={'energy':'energy_spin_so'})
    edf_spin_so = edf_spin_so.rename(columns={'cohesive':'cohesive_spin_so'})
    #     edf_spin_so = edf_spin_so.rename(columns={'formula':'formula_spin_so'})
    edf_spin_so = edf_spin_so.sort_values('formula')
    edf_spin_so = edf_spin_so.rename(columns={'elem_list':'elem_list_spin_so'}) # Added Dec 13, 2018
    edf_spin_so = edf_spin_so.rename(columns={'frac_list':'frac_list_spin_so'}) # Added Dec 13, 2018

    edf_afm_so = pd.concat((df_afm_so_Te, df_afm_so_Se, df_afm_so_S))
    edf_afm_so = edf_afm_so.rename(columns={'energy':'energy_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'cohesive':'cohesive_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'formula':'formula_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'total_elem_energy':'total_elem_energy_afm'})
    edf_afm_so = edf_afm_so.sort_values('formula_afm')
    #print('change names2')
    edf_afm_so = edf_afm_so.rename(columns={'elem_list':'elem_list_afm'}) # Added Dec 13, 2018
    edf_afm_so = edf_afm_so.rename(columns={'frac_list':'frac_list_afm'}) # Added Dec 13, 2018
    edf_afm_so = edf_afm_so.rename(columns={'elem_frac':'elem_frac_edf_afm'}) # Added Dec 13, 2018

    #     edf = pd.concat((edf_spin, edf_spin_so, edf_afm_so),axis=1)
    edf = pd.concat((edf_spin_so, edf_afm_so), axis=1)
    edf = edf.sort_values('formula')
    # edf = edf.drop(columns=['formula_spin_so','formula_afm'])#,'total_elem_energy'])
    edf = edf.drop(columns=['formula_afm'])#,'total_elem_energy'])
    #edf = edf.rename(columns={'cohesive_spin_so':'cohesive'}) # Added Dec 13, 2018
    return edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
                 df_afm_so_Te, df_afm_so_Se, df_afm_so_S]




def master_energy_df_complete(edir, main_dir, recalculate):
    """
       gathers all energy data
       returns:
           edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
                 df_spin_Te, df_spin_Se, df_spin_S,
                 df_afm_so_Te, df_afm_so_Se, df_afm_so_S], [df_nospin_Te, df_nospin_Se]
    """
    #recalculate = False

    efile_spin_Te = edir + '/energy_results_spin_Te.csv'
    efile_nospin_Te = edir + '/energy_results_nospin_Te.csv'
    efile_spin_so_Te = edir + '/energy_results_spin_so_Te.csv'
    efile_afm_so_Te = edir + '/energy_results_afm_Te.csv'
    df_nospin_Te = pd.read_csv(efile_nospin_Te, delimiter=',', usecols=[1,2])
    df_spin_Te = pd.read_csv(efile_spin_Te, delimiter=',', usecols=[1,2])
    df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
    df_afm_so_Te = pd.read_csv(efile_afm_so_Te, delimiter=',', usecols=[1,2])
    #print(df_nospin_Te.shape, df_spin_so_Te.shape, df_afm_so_Te.shape )
    # Calculate cohesive energies - Te
    df_elements = get_unique_elem_info(main_dir, df_nospin_Te, recalculate=recalculate)
    df_nospin_Te = gen_cohesive(df_nospin_Te, df_elements)
    df_spin_Te = gen_cohesive(df_spin_Te, df_elements)
    df_spin_so_Te = gen_cohesive(df_spin_so_Te, df_elements)
    df_afm_so_Te = gen_cohesive(df_afm_so_Te, df_elements)

    efile_spin_Se = edir + '/energy_results_spin_Se.csv'
    efile_nospin_Se = edir + '/energy_results_nospin_Se.csv'
    efile_afm_so_Se = edir + '/energy_results_afm_Se.csv'
    efile_spin_so_Se = edir + '/energy_results_spin_so_Se.csv'
    df_spin_Se = pd.read_csv(efile_spin_Se, delimiter=',', usecols=[1,2])
    df_nospin_Se = pd.read_csv(efile_nospin_Se, delimiter=',', usecols=[1,2])
    df_afm_so_Se = pd.read_csv(efile_afm_so_Se, delimiter=',', usecols=[1,2])
    df_spin_so_Se = pd.read_csv(efile_spin_so_Se, delimiter=',', usecols=[1,2])
    #print(df_spin_Se.shape, df_nospin_Se.shape, df_afm_so_Se.shape, df_spin_so_Se.shape )
    # Calculate cohesive energies - Se
    df_elements = get_unique_elem_info(main_dir, df_nospin_Se, recalculate=recalculate)
    df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
    df_spin_Se = gen_cohesive(df_spin_Se, df_elements)
    df_spin_so_Se = gen_cohesive(df_spin_so_Se, df_elements)
    df_afm_so_Se = gen_cohesive(df_afm_so_Se, df_elements)

    efile_spin_S = edir + '/energy_results_spin_S.csv'
    efile_nospin_S = edir + '/energy_results_nospin_S.csv'
    efile_afm_so_S = edir + '/energy_results_afm_S.csv'
    efile_spin_so_S = edir + '/energy_results_spin_so_S.csv'
    df_spin_S = pd.read_csv(efile_spin_S, delimiter=',', usecols=[1,2])
    # df_nospin_S = pd.read_csv(efile_nospin_S, delimiter=',', usecols=[1,2])
    df_afm_so_S = pd.read_csv(efile_afm_so_S, delimiter=',', usecols=[1,2])
    df_spin_so_S = pd.read_csv(efile_spin_so_S, delimiter=',', usecols=[1,2])
    #print(df_spin_S.shape,df_afm_so_S.shape,df_spin_so_S.shape)
    # Calculate cohesive energies - S
    df_elements = get_unique_elem_info(main_dir, df_spin_S, recalculate=recalculate)
    #df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
    df_spin_S = gen_cohesive(df_spin_S, df_elements)
    df_spin_so_S = gen_cohesive(df_spin_so_S, df_elements)
    df_afm_so_S = gen_cohesive(df_afm_so_S, df_elements)


    # edf_spin = pd.DataFrame()
    edf_spin = pd.concat((df_spin_Te, df_spin_Se, df_spin_S))
    edf_spin = edf_spin.rename(columns={'energy':'energy_spin'})
    edf_spin = edf_spin.rename(columns={'cohesive':'cohesive_spin'})
    # edf_spin = edf_spin.rename(columns={'formula':'formula_spin'})

    #CONCATENATE different spin ocnfigurations & relabel columns
    edf_spin_so = pd.concat((df_spin_so_Te, df_spin_so_Se, df_spin_so_S))
    edf_spin_so = edf_spin_so.rename(columns={'energy':'energy_spin_so'})
    edf_spin_so = edf_spin_so.rename(columns={'cohesive':'cohesive_spin_so'})
    edf_spin_so = edf_spin_so.rename(columns={'formula':'formula_spin_so'})

    edf_afm_so = pd.concat((df_afm_so_Te, df_afm_so_Se, df_afm_so_S))
    edf_afm_so = edf_afm_so.rename(columns={'energy':'energy_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'cohesive':'cohesive_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'formula':'formula_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'total_elem_energy':'total_elem_energy_afm'})

    edf = pd.concat((edf_spin, edf_spin_so, edf_afm_so),axis=1)
    edf = edf.drop(columns=['formula_spin_so','formula_afm'])#,'total_elem_energy'])
    return edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
                 df_spin_Te, df_spin_Se, df_spin_S,
                 df_afm_so_Te, df_afm_so_Se, df_afm_so_S], [df_nospin_Te, df_nospin_Se]





# CODES prior to changes 12.5.2018
# Not clear how much you updated this version versus (in Code/Python versus the notebook version..)
#
# #
# # df_gem.py
# #
#
# # UPDATES:
# # 9/7/2018: get_df_optima(df_master) to include Delta_FM_AFM states
#
# import os
# import pymatgen as mg
# import pymatgen as mp
#
# from math import floor, ceil
# import itertools
# import re
# import numpy as np
# import matplotlib.pyplot as plt
# import ase.db
# from mendeleev import element
# from sklearn.decomposition import PCA
# import scipy as sp
# from sklearn.preprocessing import PolynomialFeatures
# from ase.db.plot import dct2plot
# import seaborn as sns
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import cross_val_score
# from sklearn import linear_model
# from sklearn.model_selection import validation_curve
# from sklearn.linear_model import Lasso
# import random
# #mpr = MPRester("0Eq4tbAMNRe37DPs")
# from pymatgen.ext.matproj import MPRester
# m = MPRester("RK5GrTk1anSOmgAU")
#
# from scipy.stats import skew
# import shutil
#
# from fmmlcalc_b import *
# from alloy_functions import *
#
#
#
# def master_mag_df():
#     """ gather magnetic moment info for all spin configurations"""
#     edir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/energy_results'
#
#     efile_spin_Te = edir + '/magmom_results_spin_Te.csv'
#     df_spin_Te = pd.read_csv (efile_spin_Te, delimiter=',', usecols=[1,2])
#     df_spin_Te = df_spin_Te.rename(columns={'mag_mom':'magmom_spin'})
#     df_spin_Te = df_spin_Te.rename(columns={'formula':'formula_spin'})
#     #
#     efile_spin_Se = edir + '/magmom_results_spin_Se.csv'
#     df_spin_Se = pd.read_csv (efile_spin_Se, delimiter=',', usecols=[1,2])
#     df_spin_Se = df_spin_Se.rename(columns={'mag_mom':'magmom_spin'})
#     df_spin_Se = df_spin_Se.rename(columns={'formula':'formula_spin'})
#     #
#     efile_spin_S = edir + '/magmom_results_spin_S.csv'
#     df_spin_S = pd.read_csv (efile_spin_S, delimiter=',', usecols=[1,2])
#     df_spin_S = df_spin_S.rename(columns={'mag_mom':'magmom_spin'})
#     df_spin_S = df_spin_S.rename(columns={'formula':'formula_spin'})
#
#     efile_spin_so_S = edir + '/magmom_results_spin_so_S.csv'
#     df_spin_so_S = pd.read_csv (efile_spin_so_S, delimiter=',', usecols=[1,2])
#     df_spin_so_S = df_spin_so_S.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_S = df_spin_so_S.rename(columns={'formula':'formula_spin_so'})
#     #
#     efile_spin_so_Se = edir + '/magmom_results_spin_so_Se.csv'
#     df_spin_so_Se = pd.read_csv (efile_spin_so_Se, delimiter=',', usecols=[1,2])
#     df_spin_so_Se = df_spin_so_Se.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_Se = df_spin_so_Se.rename(columns={'formula':'formula_spin_so'})
#     efile_spin_so_Te = edir + '/magmom_results_spin_so_Te.csv'
#     df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
#     df_spin_so_Te = df_spin_so_Te.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_Te = df_spin_so_Te.rename(columns={'formula':'formula_spin_so'})
#
#     efile_afm_Te = edir + '/magmom_results_afm_Te.csv'
#     df_afm_Te = pd.read_csv (efile_afm_Te, delimiter=',', usecols=[1,2])
#     df_afm_Te = df_afm_Te.rename(columns={'mag_mom':'magmom_afm'})
#     df_afm_Te = df_afm_Te.rename(columns={'formula':'formula_afm'})
#     #
#     efile_afm_Se = edir + '/magmom_results_afm_Se.csv'
#     df_afm_Se = pd.read_csv (efile_afm_Se, delimiter=',', usecols=[1,2])
#     df_afm_Se = df_afm_Se.rename(columns={'mag_mom':'magmom_afm'})
#     df_afm_Se = df_afm_Se.rename(columns={'formula':'formula_afm'})
#     #
#     efile_afm_S = edir + '/magmom_results_afm_S.csv'
#     df_afm_S = pd.read_csv (efile_afm_S, delimiter=',', usecols=[1,2])
#     df_afm_S = df_afm_S.rename(columns={'mag_mom':'magmom_afm'})
#     df_afm_S = df_afm_S.rename(columns={'formula':'formula_afm'})
#     df_spin = pd.concat((df_spin_Te,df_spin_Se,df_spin_S))
#     df_spin_so = pd.concat((df_spin_so_Te,df_spin_so_Se,df_spin_so_S))
#     df_afm = pd.concat((df_afm_Te, df_afm_Se, df_afm_S))

#     df_mag_tot = pd.concat((df_spin, df_spin_so, df_afm),axis=1)
#     #formula = df_mag_tot['formula'].values
#     formula = df_spin['formula_spin'].values
#     #formula = [x[0] for x in formula]
#     #df_mag_tot = df_mag_tot.drop(df_mag_tot.columns[[4]], axis=1)
#     df_mag_tot['formula'] = formula
#     df_mag_tot = df_mag_tot.reset_index()
#     df_mag_tot = df_mag_tot.drop(columns=['index'])
#     return df_mag_tot
#

# def add_atom_counts(df_mag,mag_label, main_dir, recalculate):
#     """
# 	adds atom counts to dataframe
# 	saved to df_gen.py
#     """
#     # Calculate df_elements_mag
#     #recalculate = False
#     df_elements_mag = get_unique_elem_info(df_mag, main_dir, recalculate=recalculate)
#     #
#     total_elem_energy, elems_list_set, frac_dict_list = get_atom_energy_total(df_mag, df_elements_mag)
#     df_mag['elem_frac'] = frac_dict_list
#     df_mag['elem_list'] = elems_list_set
#     #
#     atom_label_list_mag, atom_count_list_mag = get_atom_counts(df_mag, df_elements_mag)
#     df_mag_counts = df_mag.copy()
#     for ith, atom_label in enumerate(atom_label_list_mag):
#         df_mag_counts[atom_label] = atom_count_list_mag[ith]
#     mag_mom = df_mag_counts[mag_label].values
#     if mag_mom == 'mag_mom':
#         print('Take absoulte value of magnetic moment in case it s flipped')
#         mag_mom = np.abs(mag_mom)
#     df_mag_counts.loc[:,mag_label] = mag_mom
#     return df_mag_counts

#
#
# def get_df_optima(df_master):
#     """
#         create df with nin E configurations
#         saved to df_gen.py
#     """
#     spinlabel = ['spin','spin_so','afm']
#     minE = []
#     optMag = []
#     spinlabels = []
#     spindex = []
#     #
#     delta_FM_AFM = []
#     for c in df_master[:].iterrows():
#         rownum = c[0]
#         energies = (c[1][['cohesive_spin','cohesive_spin_so','cohesive_afm']])
#         #
#         fm_energies = c[1]['cohesive_spin_so']
#         afm_energies = c[1]['cohesive_afm']
#         D_fm_afm = fm_energies - afm_energies
#         #
#         magmoms = (c[1][['magmom_spin','magmom_spin_so','magmom_afm']])
#         magmoms = np.abs(magmoms)
#         mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
#         optimum_mag = magmoms[mindex_energy]
#         optMag.append(optimum_mag)
#         minE.append(np.min(energies))
#         spindex.append(mindex_energy)
#         spinlabels.append(spinlabel[mindex_energy])
#         #
#         delta_FM_AFM.append(D_fm_afm)
#     df_optima = pd.DataFrame()
#     df_optima['formula'] = df_master['formula'].values
#     df_optima['minE'] = minE
#     df_optima['optMag'] = optMag
#     df_optima['spinlabels'] = spinlabels
#     df_optima['spindex'] = spindex
#     df_optima['delta_FM_AFM'] = delta_FM_AFM
#     return df_optima
