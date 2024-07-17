# alloy functions
import os

import pymatgen as mg
import pymatgen as mp

# creates: band_alignment.png
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


class spin_state:

    def __init__(self, atom):
        atom_info = element(atom)
        econf = atom_info.econf
        self.atom_info = atom_info
        self.econf = econf

    def parse_econf(self):
        econf = self.econf
        econf_list = econf.split(' ')
        return econf_list

    def get_orbital_info(self):
        """
            Create lists of number of states and given orbital info
            # n_vals : n orbital number
            # el_vals : l orbital number
            # ml_vals : m_l orbital number
            # Ne_state_vals : number of electrons allowed in a state. Total number of m_l
        """
        #print(econf_list, '\n')
        econf_list = self.parse_econf()
        n_vals = []
        el_vals = []
        ml_vals = []
        Ne_state_vals = []
        orbital_dict = {"s":1, "p":3, "d":5, "f":7}  # number of orbitals
	#Ne_state = # of orbitals * 2.0
        for orbital in econf_list:
            #print(orbital)
            if '[' not in orbital:
                n = orbital[0]
                el = orbital[1]
                if len(orbital) == 2:
                    #print(orbital)
                    ml = 1
                else:
                    ml = orbital[2:]
                # print(orbital)
                Ne_state = orbital_dict[el]
                n_vals.append(n)
                el_vals.append(el)
                ml_vals.append( np.float(ml) )
                # Ne_state_vals.append(Ne_state*2.0) # ERROR founf 10-4-2018
                Ne_state_vals.append(Ne_state)
                # print(n, el, ml)
                # print(Ne_state)
                # print('\n')
        # print(n_vals)
        # print(el_vals)
        # print(ml_vals)
        # print(Ne_state_vals)
        return n_vals, el_vals, ml_vals, Ne_state_vals

    def get_Nup(self):
        """
            calculate number of spin up electrons in orbital
        """
        orbital_info = self.get_orbital_info()
        Nup = 0
        l_orbitals = orbital_info[1]
        ml = orbital_info[2]
        max_num = orbital_info[3]
        #print(l_orbitals, ml, num_e)
        s_state_dex = np.argwhere(np.asarray(l_orbitals) == 's')
        if len(s_state_dex) > 0:
            s_state_dex = s_state_dex[0][0]
        else:
            s_state_dex = np.nan
        ###
        p_state_dex = np.argwhere(np.asarray(l_orbitals) == 'p')
        if len(p_state_dex) > 0:
            p_state_dex = p_state_dex[0][0]
        else:
            p_state_dex = np.nan
        ###
        d_state_dex = np.argwhere(np.asarray(l_orbitals) == 'd')
        if len(d_state_dex) > 0:
            d_state_dex = d_state_dex[0][0]
        else:
            d_state_dex = np.nan
        l_orbitals_arr = np.asarray(l_orbitals)
        f_state_dex = np.argwhere(l_orbitals_arr == "f" )
        if len(f_state_dex) > 0:
            f_state_dex = f_state_dex[0][0]
        else:
            f_state_dex = np.nan
        #print('dex', s_state_dex, d_state_dex)
        for orb_num in l_orbitals:
            #print(orb_num)
            if orb_num == 's':
                e_count = ml[s_state_dex]
                #print(s_state_dex, e_count)
                if e_count == 1:
                    Nup = Nup + 1
            if orb_num == 'p':
                e_count = np.int( ml[p_state_dex] )
                #print('econ', e_count)
                max_count = np.float(max_num[p_state_dex])
                #print('max_count', max_count)
                if e_count <= max_count:
                    Nup = Nup + e_count
                elif e_count > max_count:
                    Nup = Nup + max_count - (e_count - max_count )
            if orb_num == 'd':
                e_count = np.int( ml[d_state_dex] )
                #print('econ', e_count)
                max_count = np.float(max_num[d_state_dex])
                #print('max_count', max_count)
                if e_count <= max_count:
                    Nup = Nup + e_count
                elif e_count > max_count:
                    Nup = Nup + max_count - (e_count - max_count )
        return Nup


def gen_spin_stats(df_nna):
    """
        generate spin stats from elem_list in df
        * mean and variance of number of spin up electrons
    """
    all_elems = df_nna['elem_list']
    #print(all_elems)
    Nup_mean_list = []
    Nup_var_list = []
    for elem in all_elems:
        #print('got here')
        #print('elem : ', elem)
        Nup_list = []
        for atom in elem:
            #print(atom, type(atom))
            atom = str(atom)
            atom_spin = spin_state(atom)
            Nup = atom_spin.get_Nup()
            Nup_list.append(Nup)
        # print(Nup_list)
        Nup_mean = np.mean(Nup_list)
        Nup_var = np.var(Nup_list)
        Nup_mean_list.append(Nup_mean)
        Nup_var_list.append(Nup_var)
    return Nup_mean_list, Nup_var_list


# Nup_mean_list, Nup_var_list = gen_spin_stats(df_nna[:8])
# df_nna['elem_list'][:5]
