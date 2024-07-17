# USE onpy df_gen in Code/python

# #
# # df_gem.py
# #

# # UPDATES:
# # 9/7/2018: get_df_optima(df_master) to include Delta_FM_AFM states

# import os
# import pymatgen as mg
# import pymatgen as mp

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

# from scipy.stats import skew
# import shutil

# #from fmmlcalc_b import * needs this funciton?
# from alloy_functions import *



# def master_mag_df():
#     """ gather magnetic moment info for all spin configurations"""
#     edir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/energy_results'

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

#     efile_spin_so_S = edir + '/magmom_results_spin_so_S.csv'
#     df_spin_so_S = pd.read_csv (efile_spin_so_S, delimiter=',', usecols=[1,2])
#     df_spin_so_S = df_spin_so_S.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_S = df_spin_so_S.rename(columns={'formula':'formula_spin_so'})
#     #
#     efile_spin_so_Se = edir + '/magmom_results_spin_so_Se.csv'
#     df_spin_so_Se = pd.read_csv (efile_spin_so_Se, delimiter=',', usecols=[1,2])
#     df_spin_so_Se = df_spin_so_Se.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_Se = df_spin_so_Se.rename(columns={'formula':'formula_spin_so'})
#     #
#     efile_spin_so_Te = edir + '/magmom_results_spin_so_Te.csv'
#     df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
#     df_spin_so_Te = df_spin_so_Te.rename(columns={'mag_mom':'magmom_spin_so'})
#     df_spin_so_Te = df_spin_so_Te.rename(columns={'formula':'formula_spin_so'})

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





# def add_atom_counts(df_mag,mag_label, main_dir):
#     """
#     adds atom counts to dataframe
#     saved to df_gen.py
#     """
#     # Calculate df_elements_mag
#     recalculate = True
#     df_elements_mag = get_unique_elem_info(df_mag, main_dir, recalculate=recalculate)
#     #
#     total_elem_energy, elems_list_set, frac_dict_list = get_atom_energy_total(df_mag, df_elements_mag)
#     df_mag['elem_frac'] = frac_dict_list
#     df_mag['elem_list'] = elems_list_set
#     #
#     atom_label_list_mag, atom_count_list_mag = get_atom_counts(df_mag, df_elements_mag)
#     df_mag_counts = df_mag.copy(deep = True)
#     for ith, atom_label in enumerate(atom_label_list_mag):
#         df_mag_counts[atom_label] = atom_count_list_mag[ith]

#     mag_mom = df_mag_counts[mag_label].values
#     mag_mom = np.abs(mag_mom)
#     df_mag_counts.loc[:,mag_label] = mag_mom
#     return df_mag_counts



# def get_df_optima(df_master):
#     """
#         Create df with min E configurations
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
#         print('energies',energies)
#         fm_energies = c[1]['cohesive_spin_so']
#         # print(fm_energies)
#         afm_energies = c[1]['cohesive_afm']
#         D_fm_afm = fm_energies - afm_energies
#         #
#         magmoms = (c[1][['magmom_spin','magmom_spin_so','magmom_afm']])
#         magmoms = np.abs(magmoms)
#         mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
#         print('mindex_energy',mindex_energy)
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
