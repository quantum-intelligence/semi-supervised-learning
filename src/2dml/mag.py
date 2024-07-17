# alloy function

#
# Resaved from jupyterntebook CGT_ML_tf_v17_vector_p on Sept 17, 2018
#

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

from scipy.stats import skew
import shutil
#from df_gen import *
from fmmlcalc_b import *



def get_unique_elem_info(main_dir, df, recalculate=False):
    """
        Get all possible elements first, get unique list and call
        materials project using this..
    """
    cwd = os.getcwd()
    os.chdir(main_dir)
    if not os.path.exists('elements_energies.csv') or recalculate:
        elems_list = []
        for ith, cmpd in df['formula'].iteritems():
            #if ith < 1:
            #cmpd_list.append(cmpd)
            mp_cmpd = mp.Composition(cmpd)
            elems = mp_cmpd.elements
            for el in elems:
                elems_list.append(el)
        #get unique elements and extract info from materials project:
        # change to list of strings so works with np.unique()
        elems_list = [str(x) for x in elems_list]
        unique_elem = np.unique(elems_list)
        elem_min_energy = []
        for elem in unique_elem:
            if elem == 'Ti':
                m_min_energy = -7.833
            elif elem == 'Ni':
                m_min_energy = -5.468
            elif elem == 'Cu':
                m_min_energy = -3.728
            elif elem == 'Nb':
                m_min_energy = -10.215
            elif elem == 'Mo':
                m_min_energy = -10.934
            else:
                m_elem = m.get_data(elem)
                m_energies = [x['energy_per_atom'] for x in m_elem]
                m_min_energy = np.min(m_energies)
            elem_min_energy.append(m_min_energy)

        df_elements = pd.DataFrame()
        df_elements['unique_elem'] = unique_elem
        df_elements['elem_min_energy'] = elem_min_energy

        df_elements.to_csv('elements_energies.csv')
    else:
        df_elements = pd.read_csv('elements_energies.csv',delimiter=',',usecols=[1,2])
    os.chdir(cwd)
    return df_elements




from pymatgen.io.vasp.inputs import Poscar



def make_alloy(CGT_, target_atom, replace_atom, site, allsites):
    """
        makes alloy of CGT
        * replace one target atom with another.
        * 50%/50% alloy
    """
    CGT = CGT_.copy()
    #target_atom = 'Cr'
    #replace_atom = 'Co'
    t_sites = [s for s in enumerate(CGT.sites) if s[1].specie == mg.Element(target_atom) ]
    #print 't_sites :  ', t_sites, '\n'
    #print t_sites[0]
    site_num = t_sites[site][0] #take first site , site number
    #print 'site_num ', site_num
    if allsites == True:
        CGT.replace_species({target_atom:replace_atom})
    else:
        CGT.replace(site_num,replace_atom)
    return CGT

def gen_pairlist(atoms_list):
    """ generate pairs from list of atoms """
    pair_list = []
    # B_atoms = ['Ge','Si','P']
    B_atoms = atoms_list
    for ith, i in enumerate(B_atoms):
        for jth, j in enumerate(B_atoms):
            if ith <= jth:
                pair = [i,j]
                pair_list.append(pair)
    return pair_list

def make_alloy_pair(CGT_, target_atom, pair):
    """ make allow wiht pair or replacement atoms as input """
    #pair = ['Si','Sn']
    replace_atom = pair[0]
    site = 0
    CGT = CGT_.copy()
    t_sites = [s for s in enumerate(CGT.sites) if s[1].specie == mg.Element(target_atom) ]
    #print 't_sites :  ', t_sites, '\n'
    #print t_sites[0]
    site_num = t_sites[site][0] #take first site , site number
    CGT.replace(site_num,replace_atom)
    replace_atom = pair[1]
    CGT.replace(site_num+1,replace_atom)
    return CGT


def alloy_testdata_gen(CGT, X_atoms, B_atoms, replace_atoms ):
    """
        generates alloy data
    """
    CGT_ = CGT.copy()
    abx_alloys = []
    pair_list = gen_pairlist(B_atoms)
    writefiles = True

    #dir_stem = os.getcwd()
    dir_stem = u'/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/test_set/'
    print('dir stem', dir_stem)
    for X_atom in X_atoms:
        targetX_atom = 'Te'
        allsites = True
        site = 0
        ith_alloy = make_alloy(CGT_, targetX_atom, X_atom, site, allsites)

        for pair in pair_list:
            target_atom = 'Ge'
            jth_alloy = make_alloy_pair(ith_alloy, target_atom, pair)
            for replace_atom in replace_atoms:
                allsites = False
                target_atom = 'Cr'
                kth_alloy = make_alloy(jth_alloy, target_atom, replace_atom, site, allsites)
                abx_alloys.append(kth_alloy)

                kth_formula = kth_alloy.formula
                kth_formula = kth_formula.replace(' ','_')
                temp_filename = kth_formula + '.cif'
                if writefiles == True:
                    print('kth_formula', kth_formula)
                    directory = 'alloys/'+ kth_formula
                    print('directory: ', directory)
                    directory = directory.replace('_','')
                    print('directory: ', directory)
                    if not os.path.exists(directory):
                        print('creating directory... ')
                        os.makedirs(directory)
                    else:
                        print(directory, ' exists already')
                    poscar_kth = 'POSCAR_' + kth_formula
                    make_poscar = Poscar(kth_alloy)
                    #str_poscar = make_poscar.get_string(direct=False)
                    os.chdir(directory)
                    make_poscar.write_file(poscar_kth, direct=False)
                    #dst = directory + '/' + 'POSCAR' #+ '_' + kth_formula
                    shutil.copy(poscar_kth, 'POSCAR')
                    os.chdir(dir_stem)
    return abx_alloys




def get_atom_counts(df, df_elements, elem_list):
    """
        gets atom counts from df and df_elements
        returns list of atom labels and columns of data
        for use with adding to a dataframe
    """
    df_analyze =  df.dropna(axis=0).copy()
    # df_analyze[df_analyze['cohesive'] > -0.4]
    unique_elements = df_elements['unique_elem'].values
    # create vector of length unique_elements
    N_ELEM = len(unique_elements)
    N = df.shape[0]

    atom_matrix = np.zeros((N_ELEM,N))
    for c, cmpd in df[elem_list].iteritems():
        if 'elem_frac' in df.columns:
            frac_dict = df['elem_frac'][c]
        else:
            frac_dict = df['elem_frac_edf'][c]
        atom_vector = np.zeros(len(unique_elements))
        print(frac_dict)
        print(cmpd)
        for ath, atom in enumerate(cmpd):
            print(atom) 
            atomic_ratio = frac_dict[ath][atom]*10.0
            # print('ratio', atomic_ratio)
            # get index in unique_elements and use for atom_vector
            atom_index = np.argwhere(unique_elements == str(atom))[0][0]
            # print(atom, atom_index)
            atom_vector[atom_index] = atomic_ratio
        atom_matrix[:,c] = atom_vector
    atom_count_list = []
    atom_label_list = []
    for i in np.arange(N_ELEM):
        atom_column = atom_matrix[i]
        atom_count_list.append(atom_column)
        atom_label_list.append(unique_elements[i])
    return atom_label_list, atom_count_list




def Bexists(i, B_atom_pair, df_counts):
    """
        find if pair of B atoms exists
        and use to determine whether to pick up energy term
        for Matrix of B atoms and TM atoms
        - this functino may be altering original df despite df_counts being separated by many copy)( calls)
    """
    df_counts1 = df_counts.copy(deep=True)
    sgp = [mg.Element(x) for x in B_atom_pair] #sgp --> Si, Ge, P
    #tm = [mg.Element(x) for x in tm]
    cmpd = df_counts1['formula'][i]
    frac = df_counts1['elem_frac'][i]
    # elem_list = df_counts1['elem_list'][i]
    elem_list = df_counts1['elem_list'].values[i]
    #print('formula',cmpd)
    #print('elem_list', elem_list)
    #print('frac', frac)
    frac_list = []
    for fth, f in enumerate(frac):
        #print(fth, f, elem_list[fth])
        atom_count = f[elem_list[fth]]
        #print(atom_count)
        frac_list.append(atom_count)
    if type(elem_list) == list:
        elem_list_copy = elem_list[:] #slicing created a clone or copy
    else:
        elem_list_copy = elem_list.copy(deep=True)
    keye = [False, False]
    for ath, atom in enumerate(sgp):
        #print('BEFORE IF STATE', type(atom),atom, elem_list_copy)
        if atom in elem_list_copy:
            keye[ath] = True
            #print('here', keye)
            elem_str = [str(x) for x in elem_list_copy]
            #print('ELEMLIST', elem_str, atom, type(atom))
            atom_index = np.where(str(atom) == np.array(elem_str))[0][0]
            #print(atom_index)
            stoic = frac_list[atom_index]*10.0
            #print('atom',atom_index, stoic)
            if (stoic == 2) & (sgp[0] == sgp[1]):
                #print('got here')
                return True
            elem_list_copy.remove(atom)
    return all(keye)


def TMexists(i, TM_atom, df_counts):
    """
        find if TM atoms exists
        and use to determine whether to pick up energy term
        for Matrix of TM atoms
    """
    istm = False
    tm = mg.Element(TM_atom)
    cmpd = df_counts['formula'][i]
    frac = df_counts['elem_frac'][i]
    elem_list = df_counts['elem_list'][i]
    #print('elem_list', elem_list)
    frac_list = []
    #print('FRAC',frac)
    for fth, f in enumerate(frac):
        #print(fth, f, elem_list[fth])
        atom_count = f[elem_list[fth]]
        frac_list.append(atom_count)
    #print(tm, type(tm))
    if tm in elem_list:
        #print('here', keye)
        elem_str = [str(x) for x in elem_list]
        #print('ELEMLIST', elem_str, atom, type(atom))
        atom_index = np.where(str(tm) == np.array(elem_str))[0][0]
        stoic = frac_list[atom_index]*10.0
        istm = True
    return istm



def EnergyMatrixGen_OLD(df_counts,TMlist,B_atom_pair):
    """ generates EnergyMAtrix from B list, TM list
        constrain for Te containing, or Se containing atoms etc prior
    """
    df_counts1 = df_counts.copy(deep=True)
    df_counts1 = df_counts1.reset_index()
    EnergyMatrix = np.ones((len(TMlist),len(B_atom_pair)))
    #print(EnergyMatrix)
    for i,cmpd in enumerate(df_counts1['formula'][:]):
        for bth, b in enumerate(B_atom_pair):
            for tmth, tm in enumerate(TMlist):
                #print(i,bth,tmth)
                Btrue = Bexists(i, b, df_counts1)
                TMtrue = TMexists(i, tm, df_counts1)
                if Btrue and TMtrue:
                    TMB_energy = df_counts1['cohesive'][i]
                    EnergyMatrix[tmth,bth] = TMB_energy

    EnergyMatrix_nna = EnergyMatrix.copy()
    EnergyMatrix_nna[np.isnan(EnergyMatrix_nna)] = 1.0  ## NOT GOOD!!!
    return EnergyMatrix_nna




def element_energy_lookup(atom,df_elements):
    """
        look up energy of element using df_elements dataframe
    """
    df_elements_type = type(df_elements['unique_elem'][0])
    if df_elements_type == str:
        if type(atom) == str: #changes to string when read in from csv, mp.Element otherwise
            #print('atom type strying', atom, type(atom))
            target_row = df_elements.loc[df_elements['unique_elem'] == atom]
        else:
            #print('atom type not string44')
            target_row = df_elements.loc[df_elements['unique_elem'] == str(atom)]
    else:
        if type(atom) == str: #changes to string when read in from csv, mp.Element otherwise
            #print('atom type strying', atom, type(atom))
            target_row = df_elements.loc[df_elements['unique_elem'] == mp.Element(atom)]
        else:
            #print('atom type not string44')
            target_row = df_elements.loc[df_elements['unique_elem'] == atom]
    #print('atom', atom)
    #print('target_row',target_row)
    # if len(target_row) == 0:
    #     print('empty target row')
    #     print(df_elements)
    #     print(atom)
    target_energy = target_row['elem_min_energy'].values[0]
    # print(target_energy)
    return target_energy


def get_atom_energy_total(df, df_elements):
    """
        calculate list of sum of energies of constituent atoms of compounds
    """
    elems_list_set = []
    total_elem_energy = []
    frac_dict_list = []
    for ith, cmpd in df['formula'].iteritems():
        mp_cmpd = mp.Composition(cmpd)
        num_atoms = mp_cmpd.num_atoms
        elems = mp_cmpd.elements
        elems_list = []
        frac_list = []
        fraction_dict = []
        energy_dict = []
        energy_list = []
        for el in elems:
            elems_list.append(el)
            elem_frac = mp_cmpd.get_atomic_fraction(el)
            frac_list.append(elem_frac)
            elem_frac_dict = {el : elem_frac}
            energy = element_energy_lookup(el, df_elements)
            energy_list.append(energy)
            elem_energy = {el : energy}
            fraction_dict.append(elem_frac_dict)
            energy_dict.append(elem_energy)
        frac_list = np.asarray(frac_list)
        energy_frac_ratios = energy_list*np.asarray(frac_list)*num_atoms
        energy_sum = np.sum(energy_frac_ratios)
        total_elem_energy.append(energy_sum)
        # collect elements info
        elems_list_set.append(elems_list)
        frac_dict_list.append(fraction_dict)
    return total_elem_energy, elems_list_set, frac_dict_list



def shells_stats(shell):
    """
    Takes the last subshell given by mendeleev package and returns
        The sum of 2p and d electrons given by the formula.
        What about # electrons within the unit cell?
        - updated to get only 'p' electrons
        - updaed for division by zero error!! 12.22.2018 for d and f electrions
    """
    cmpd_p = []
    cmpd_d = []
    cmpd_f = []
    cmpd_skew_p = []
    cmpd_skew_d = []
    cmpd_skew_f = []
    cmpd_sigma_p = []
    cmpd_sigma_d = []
    cmpd_sigma_f = []
    for l in shell:
        N = len(l)
        num_p = [0]
        num_d = [0]
        num_f = [0]
        for ith in l:
            if ith[0][1] == 'p':
                num_p.append(ith[1])
            if ith[0][1] == 'd':
                num_d.append(ith[1])
            if ith[0][1] == 'f':
                num_f.append(ith[1])
        num_p = np.asarray(num_p)
        num_d = np.asarray(num_d)
        num_f = np.asarray(num_f)
        mean_num_p = np.mean(num_p)
        mean_num_d = np.mean(num_d)
        mean_num_f = np.mean(num_f)
        if mean_num_p == 0:
            sigma_p = 0
        else:
            sigma_p = np.var(num_p)/(np.mean(num_p))**2.0
        if mean_num_d == 0:
            sigma_d = 0
        else:
            sigma_d = np.var(num_d)/(np.mean(num_d))**2.0
        if mean_num_f == 0:
            sigma_f = 0
        else:
            sigma_f = np.var(num_f)/(np.mean(num_f))**2.0
        sum_p = np.sum(num_p)
        sum_d = np.sum(num_d)
        sum_f = np.sum(num_f)
        skew_p = skew(num_p)
        skew_d = skew(num_d)
        skew_f = skew(num_f)
        cmpd_p.append(sum_p)
        cmpd_d.append(sum_d)
        cmpd_f.append(sum_f)
        cmpd_skew_p.append(skew_p)
        cmpd_skew_d.append(skew_d)
        cmpd_skew_f.append(skew_f)
        cmpd_sigma_p.append(sigma_p)
        cmpd_sigma_d.append(sigma_d)
        cmpd_sigma_f.append(sigma_f)
    return (cmpd_p, cmpd_d, cmpd_f,
            cmpd_skew_p, cmpd_skew_d, cmpd_skew_f,
            cmpd_sigma_p, cmpd_sigma_d, cmpd_sigma_f)




def build_ABX_mendel(df1, atomicrad, atomicvol, covalentrad, dipole, eaffinity,
                     numelectron, ionenergies, oxi, vdwradius, en, nvalence, elem_list,
                     weights, lastshell, boiling_point, density, evaporation_heat,
                     fusion_heat, gas_basicity,  heat_of_formation, melting_point,
                     thermal_conductivity):
    """
        Found in alloy_functions.py
        build up df using mendeleev data

    """
    (num_p, num_d, num_f, cmpd_skew_p, cmpd_skew_d, cmpd_skew_f,
     cmpd_sigma_p, cmpd_sigma_d, cmpd_sigma_f) = shells_stats(lastshell)

    df1['num_p'] = num_p
    df1['num_d'] = num_d
    df1['num_f'] = num_f
    df1['atomic_rad'] = atomicrad
    df1['atomic_vol'] = atomicvol
    df1['covalent_rad'] = covalentrad
    df1['dipole'] = dipole
    df1['eaffinity'] = eaffinity
    df1['num_electrons'] = numelectron

    #property_values = nvalence
    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(atomicrad)
    df1['atomic_rad_sum_dif'] = sum_dif_list
    df1['atomic_rad_std_dif'] = std_dif_list
    df1['atomic_rad_std'] = prop_std_list
    df1['atomic_rad_avg'] = weighted_avg(atomicrad,weights)
    df1['atomic_rad_max_dif'] = max_difference(atomicrad)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(atomicvol)
    df1['atomic_vol_sum_dif'] = sum_dif_list
    df1['atomic_vol_std_dif'] = std_dif_list
    df1['atomic_vol_std'] = prop_std_list
    df1['atomic_vol_avg'] = weighted_avg(atomicvol,weights)
    df1['atomic_vol_max_dif'] = max_difference(atomicvol)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(covalentrad)
    df1['covalentrad_sum_dif'] = sum_dif_list
    df1['covalentrad_std_dif'] = std_dif_list
    df1['covalentrad_std'] = prop_std_list
    df1['covalentrad_avg'] = weighted_avg(covalentrad, weights)
    df1['covalentrad_max_dif'] = max_difference(covalentrad)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(dipole)
    df1['dipole_sum_dif'] = sum_dif_list
    df1['dipole_std_dif'] = std_dif_list
    df1['dipole_std'] = prop_std_list
    df1['dipole_avg'] = weighted_avg(dipole, weights)
    df1['dipole_max_dif'] = max_difference(dipole)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(eaffinity)
    df1['eaffinity_sum_dif'] = sum_dif_list
    df1['eaffinity_std_dif'] = std_dif_list
    df1['eaffinity_std'] = prop_std_list
    df1['e_affinity_avg'] = weighted_avg(eaffinity, weights)
    df1['e_affinity_max_dif'] = max_difference(eaffinity)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(numelectron)
    df1['numelectron_sum_dif'] = sum_dif_list
    df1['numelectron_std_dif'] = std_dif_list
    df1['numelectron_std'] = prop_std_list
    df1['numelectron_avg'] = weighted_avg(numelectron,weights)
    df1['numelectron_max_dif'] = max_difference(numelectron)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(vdwradius)
    df1['vdwradius_sum_dif'] = sum_dif_list
    df1['vdwradius_std_dif'] = std_dif_list
    df1['vdwradius_std'] = prop_std_list
    df1['vdwradius_avg'] = weighted_avg(vdwradius,weights)
    df1['vdwradius_max_dif'] = max_difference(vdwradius)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(en)
    df1['e_negativity_sum_dif'] = sum_dif_list
    df1['e_negativity_std_dif'] = std_dif_list
    df1['e_negativity_std'] = prop_std_list
    df1['e_negativity_avg'] = weighted_avg(en,weights)
    df1['e_negativity_max_dif'] = max_difference(en)

    sum_dif_list, std_dif_list, prop_std_list = dif_stats_calc(nvalence)
    df1['nvalence_sum_dif'] = sum_dif_list
    df1['nvalence_std_dif'] = std_dif_list
    df1['nvalence_std'] = prop_std_list
    df1['nvalence_avg'] = weighted_avg(nvalence,weights)
    df1['nvalence_max_dif'] = max_difference(nvalence)

    df1['lastsubshell_avg'] = lastshell
    #print('lastshell',len(lastshell))
    #print('cmpd_skew_p', len(cmpd_skew_p))
    df1['cmpd_skew_p'] = cmpd_skew_p
    df1['cmpd_skew_d'] = cmpd_skew_d
    df1['cmpd_skew_f'] = cmpd_skew_f
    df1['cmpd_sigma_p'] = cmpd_sigma_p
    df1['cmpd_sigma_d'] =  cmpd_sigma_d
    df1['cmpd_sigma_f'] =  cmpd_sigma_f

    # include atomization energy of first and second atom
    atomE_AB = [x[1]+x[0] for x in heat_of_formation]
    df1['atomE_AB'] = atomE_AB #atomE_A + atomE_B
    Total_e = get_eTot(numelectron)
    frac_f = num_f/Total_e*1.0
    df1['frac_f '] = frac_f

    ion_list, std_ion, sum_ion, mean_ion = get_ionization_stats(ionenergies)
    df1['std_ion'] = std_ion
    df1['sum_ion'] = sum_ion
    df1['mean_ion'] = mean_ion
    # Add Born-Haber term
    # Born = dif_ion - max_difference(eaffinity) #this doesn't work for some reason
    Born = df1['std_ion'].values - df1['eaffinity_std'].values + df1['atomE_AB'] .values
    df1['Born'] = Born
    return df1




def dif_stats_calc(property_values):
    """
        calculate sum of differences and the
        variamce (not standard deviation) of the differences
        - scintillation instead of variance..
    """
    sum_dif_list = []
    std_dif_list = []
    prop_std_list = []
    for n, val in enumerate(property_values):
        prop = property_values[n]
        prop = [x for x in prop if x != None]
        prop_std = np.var(np.asarray(prop))/(np.mean(np.asarray(prop)))**2.0
        #print('prop', property_values, prop)
        dif = np.zeros((len(prop), len(prop)))
        #print(dif.shape)
        for i in np.arange(len(prop)):
            for j in np.arange(len(prop)):
                if i != j:
                    #print('prop[i]',prop[i])
                    dif[i,j] = np.abs(prop[i] - prop[j])
        if len(prop) == 3:
            prefactor = 8.0/3.0
        elif len(prop) == 4:
            prefactor = 3.0/2.0
        elif len(prop) > 4:
            prefactor = 1.0
        #print(len(prop))
        sum_dif = prefactor*np.sum(0.5*dif)  # account for double counting
        dif_std = np.var(dif)/(np.mean(dif))**2.0
        #print(sum_dif, sum_std)
        sum_dif_list.append(sum_dif)
        std_dif_list.append(dif_std)
        prop_std_list.append(prop_std)
    return sum_dif_list, std_dif_list, prop_std_list




def get_ionization_stats(ionenergies):
    """
        get first three ionization energies. OR first if Hydrogen
    """
    entry_list = []
    std_list = []
    sum_list = []
    mean_list = []
    for ith, entry in enumerate(ionenergies):
        #print ith
        ion_list = []
        for ion in entry:
            #take the mean of the first three if big enough!
            if len(ion)>1:
                #first_ion = (ion[1]+ion[2]+ion[3])/3.0
                #print ion[0]
                first_ion = (ion[1]+ion[2])/3.0
            else:
                first_ion = ion[1]
            #print ion[1]
            ion_list.append(first_ion)
        #print ion_list
        ion_list = np.asfarray(ion_list)
        std_ion = np.std(ion_list)
        sum_ion = np.sum(ion_list)
        mean_ion = np.mean(ion_list)
        entry_list.append(ion_list)
        std_list.append(std_ion)
        sum_list.append(sum_ion)
        mean_list.append(mean_ion)

    return entry_list, std_list, sum_list, mean_list



def calc_hardness_stats(elements_set):
    """
        calculates chemical hardness
    """
    prop_stats_set = []
    for ith, species in enumerate(elements_set):
        # print(species)
        prop_stats = propertyStats(species, 11) #11 for hardness
        prop_stats_set.append(prop_stats)
    prop_stats_set = np.asarray(prop_stats_set)
    hard_mean = prop_stats_set[:,0]
    hard_var = prop_stats_set[:,1]
    return hard_mean, hard_var



#import pymatgen as mg
from pymatgen import Lattice, Structure
from mendeleev import element


def get_ionic_crystal_r(mdlv):
    """  gets 'most reliable'  """
    # len(element('Cr').ionic_radii)
    #mdlv = element(str(elem))
    reliable = [x.most_reliable for x in mdlv.ionic_radii]
    #print(reliable)
    #print((np.asarray(reliable) == True).any())
    # Fin position of 'most relieavle' estimate, otherwise just pick top one if none listed
    if (np.asarray(reliable) == True).any() == True:
        rdex = np.argwhere(np.asarray(reliable) == True)[0][0]
        #print(rdex)
        crystal_radius = element('Cr').ionic_radii[rdex].crystal_radius
        ionic_radius = element('Cr').ionic_radii[rdex].ionic_radius
        element('Cr').ionic_radii[rdex]
    else:
        rdex = 0
        #print(rdex)
        crystal_radius = element('Cr').ionic_radii[rdex].crystal_radius
        ionic_radius = element('Cr').ionic_radii[rdex].ionic_radius
        element('Cr').ionic_radii[rdex]
    return ionic_radius, crystal_radius


def property_vector_gen(species):
    """
        parses list of mendeleev elements to vectors of their
        corresponding atomic properties
    """
    atomic_number_list = []
    atomic_radius_list = []
    atomic_radius_rahm_list = []
    covalent_radius_pyykko_list = []
    dipole_polarizability_list = []
    electron_affinity_list = []
    en_allen_list = []
    gas_basicity_list = []
    crystal_radius_list = []
    ionic_radius_list = []
    nvalence_list = []
    hardness_list = []
    #
    for elem in species:
        #print(elem)
        element(str(elem))
        # elem = species[0]
        mdlv = element(str(elem))
        m_atomic_number = mdlv.atomic_number
        m_atomic_radius = mdlv.atomic_radius
        m_atomic_radius_rahm = mdlv.atomic_radius_rahm
        m_covalent_radius_pyykko = mdlv.covalent_radius_pyykko
        m_dipole_polarizability = mdlv.dipole_polarizability
        m_electron_affinity = mdlv.electron_affinity
        m_en_allen = mdlv.en_allen
        m_gas_basicity = mdlv.gas_basicity
        m_ionic_radius, m_crystal_radius = get_ionic_crystal_r(mdlv)
        m_nvalence = mdlv.nvalence()
        m_hardness = mdlv.hardness()
        # m_crystal_radius = mdlv.crystal_radius
        # m_ionic_radius = mdlv.ionic_radius
        atomic_number_list.append( m_atomic_number )
        atomic_radius_list.append( m_atomic_radius )
        atomic_radius_rahm_list.append( m_atomic_radius_rahm )
        covalent_radius_pyykko_list.append( m_covalent_radius_pyykko )
        dipole_polarizability_list.append( m_dipole_polarizability )
        electron_affinity_list.append( m_electron_affinity )
        en_allen_list.append( m_en_allen )
        gas_basicity_list.append( m_gas_basicity )
        crystal_radius_list.append( m_gas_basicity )
        ionic_radius_list.append( m_ionic_radius )
        nvalence_list.append(m_nvalence)
        hardness_list.append(m_hardness)
        results_list = [atomic_number_list, atomic_radius_list, atomic_radius_rahm_list,
                        covalent_radius_pyykko_list, dipole_polarizability_list, electron_affinity_list,
                        en_allen_list, gas_basicity_list, crystal_radius_list,
                        ionic_radius_list, nvalence_list, hardness_list ]
    return results_list



def propertyStats(species, p_num):
    """
       Create matrix from vector of property_vector_gen...
       let's say covalent_radius_pyykko
    """
    #p_num = 3
    p_vec = property_vector_gen(species)[p_num]
    # p_vec ==> covalent_radius_p_vec
    p_vec = [x for x in p_vec if x != None]
    #print('pvec', p_vec)
    #print('species',species)
    mean_p = np.mean(p_vec)
    var_p = np.var(p_vec)
    #print('mean, var', mean_p, var_p)
    return mean_p, var_p



def propertyMatrix(species, p_num):
    """
       Create matrix from vector of property_vector_gen...
       let's say covalent_radius_pyykko
    """
    #p_num = 3
    p_vec = property_vector_gen(species)[p_num]
    # p_vec ==> covalent_radius_p_vec

    N = len(p_vec)
    deltar = np.zeros((N,N))
    for i in np.arange(len(p_vec)):
        for j in np.arange(len(p_vec)):
            #print(i,j)
            #print(p_vec[i])
            deltar[i,j] = (p_vec[i] + p_vec[j])
    return deltar


def property_dif_Matrix(species, p_num):
    """
       Create matrix from vector of property_vector_gen...
       let's say covalent_radius_pyykko
    """
    #p_num = 3
    p_vec = property_vector_gen(species)[p_num]
    # p_vec ==> covalent_radius_p_vec

    N = len(p_vec)
    deltar = np.zeros((N,N))
    for i in np.arange(len(p_vec)):
        for j in np.arange(len(p_vec)):
            #print(i,j)
            #print(p_vec[i])
            deltar[i,j] = np.abs(p_vec[i] - p_vec[j])
    return deltar

def muProxy_gen(distmatrix, pMat, nval, ABweight):
    """
        connect distanceMatrix with propertyMatrix
    """
    gamma = 0.01 #imaginary number?
    muProxy = distmatrix/pMat*( 1.0 - 1.0/(nval + gamma) )
    muProxy[:36,:36] = muProxy[:36,:36]*ABweight
    muProxy = muProxy/ABweight
    #vectorize muProxy matrix
    dimN = muProxy.shape[0]
    muProxy_vec = muProxy.reshape((dimN*dimN,1))
    return muProxy_vec


def chem_sim_gen(muProxy_vec_set):
    """
        # define chemical similarity:
    """
    Lnorm = 1.0
    zeroref = False
    chemsim_vec = []
    for ith, vec in enumerate(muProxy_vec_set):
        if zeroref == True:
            vec_ref = np.zeros(len(vec))
            #print('ith', ith)
            chem_sim = np.sum((vec - vec_ref)**Lnorm)
            chemsim_vec.append(chem_sim)
        else:
            if ith == 0:
                vec_ref = vec
                chem_sim = 0.0
                chemsim_vec.append(chem_sim)
            else:
                #print(ith)
                diff = (np.abs(vec - vec_ref))**Lnorm
                #
                chem_sim = np.sum(diff)
                chemsim_vec.append(chem_sim)
                #print(chem_sim)
    return chemsim_vec


# ext = '/Users/trevorrhone/Documents/Kaxiras/2DML/Shaan/testdir/'

def parse_contcar_data(ext, contdir):
    """
        extract contar data from folder
    """
    dirs = os.listdir(ext)
    #contdir = os.listdir(ext)
    #print('here', contdir)
    #contdir = [x for x in contdir if 'contcar' in x][0]
    #contdir = ext + contdir
    dirs = os.listdir(ext+'/'+contdir)
    dirs = [x for x in dirs if '.' not in x]
    dirs = [x for x in dirs if 'contcar_folder' not in x]
    dirs = [x for x in dirs if '__' not in x]
    #print('DORS', dirs)

    mg_structures = []
    formula = []
    for dir in dirs:
        cmpd_name = dir.split('_')[0]
        formula.append(cmpd_name)
        structure_label = ext + contdir + '/' + dir #+ '_CONTCAR'
        mg_structure = Structure.from_file(structure_label)
        #
        # REMOVE some X sites (eg Te)
        # mg_structure.remove_sites([5,6,8,9])  #KEEP ALL X sites
        #
        #mg_structure.make_supercell(2.0)
        mg_structures.append(mg_structure)

    distmatrix_set = []
    elements_set =[]
    for struc in mg_structures:
        #print(struc)
        distmatrix = (struc.distance_matrix)
        distmatrix = np.abs(distmatrix) #ensure all entries are positibve
        sites = struc.sites
        species = [x.specie for x in sites]
        #print(species)
        distmatrix_set.append(distmatrix)
        elements_set.append(species)
    return elements_set, distmatrix_set, formula


def calc_chem_similarity(elements_set, distmatrix_set,p_num,ABweight):
    """
        calculates chemical similarity
    """
    #p_num = 9  # 3 for covalent_radius
    muProxy_vec_set = []
    prop_stats_set = []
    for ith, species in enumerate(elements_set):
        #print(species)
        distmatrix = distmatrix_set[ith]
        pMat = propertyMatrix(species, p_num)
        nval = property_dif_Matrix(species, 10)
        muProxy_vec = muProxy_gen(distmatrix, pMat,nval, ABweight)
        muProxy_vec_set.append(muProxy_vec)

    chemsim_vec = chem_sim_gen(muProxy_vec_set)
    return chemsim_vec #, prop_stats_set


def df_delta( df_spin_Te, df_nospin_Te, label, elabel1, elabel2 ):
    """
        get difference in df[energies] to deternine the spin state
        - updated df_delta to include ability to either copy elem_frac columns
        from inputs or to generate it on the fly if absent
    """
    delta_spin = df_spin_Te[elabel1].values - df_nospin_Te[elabel2].values
    df_spin_polariz_Te = pd.DataFrame()
    df_spin_polariz_Te['formula'] = df_nospin_Te['formula'].values
    df_spin_polariz_Te['energy_dif'] = delta_spin
    df_spin_polariz_Te[label] = 0
    spindex = ([df_spin_polariz_Te['energy_dif'] < 0])
    spin_nandex = pd.isnull(df_spin_polariz_Te['energy_dif'])
    spin_states = df_spin_polariz_Te[label].values
    spin_states[spindex] = 1
    spin_states[spin_nandex] = 3.142
    df_spin_polariz_Te[label] = spin_states
    if 'elem_list' in df_spin_Te.columns: #'elem_list', 'elem_frac',
        df_spin_polariz_Te['elem_frac'] = df_spin_Te['elem_frac'].values
        df_spin_polariz_Te['elem_list'] = df_spin_Te['elem_list'].values
    else:
        df_spin_polariz_Te = add_atom_counts(df_spin_polariz_Te)
    return df_spin_polariz_Te


def df_deltaOLD( df_spin_Te, df_nospin_Te, label, elabel1,elabel2 ):
    """
        get difference in df[energies] to deternine the spin state
    """
    delta_spin = df_spin_Te[elabel1].values - df_nospin_Te[elabel2].values
    df_spin_polariz_Te = pd.DataFrame()
    df_spin_polariz_Te['formula'] = df_nospin_Te['formula'].values
    df_spin_polariz_Te['energy_dif'] = delta_spin
    df_spin_polariz_Te[label] = 0
    spindex = ([df_spin_polariz_Te['energy_dif'] < 0])
    spin_nandex = pd.isnull(df_spin_polariz_Te['energy_dif'])
    spin_states = df_spin_polariz_Te[label].values
    spin_states[spindex] = 1
    spin_states[spin_nandex] = 3.142
    df_spin_polariz_Te[label] = spin_states
    return df_spin_polariz_Te


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
