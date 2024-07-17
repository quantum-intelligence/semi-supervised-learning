import sys
sys.path.append("/Users/trevorrhone/Documents/Code/Python/2dml")
# sys.path.append("./codes") #hpc
import numpy as np
np.random.seed(0)
import pymatgen as mp
# import tensorflow as tf
# tf.compat.v1.random.set_random_seed(0)
# from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Model
# from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from mpl_toolkits.mplot3d import Axes3D
from mendeleev import element
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from importlib import reload
# import keras
from ase import Atoms
import ase.db as ad

from alloy_functions import *
# from NN_codes import *

import pickle
# import django

import ase
import ase.io
import os
from dscribe.descriptors import SOAP
from pymatgen.core.composition import Composition
import pymatgen.core as mp


from pymatgen.io.ase import AseAtomsAdaptor

import gen_atomic_descriptors
reload(gen_atomic_descriptors)
from gen_atomic_descriptors import *


def gen_unique_formulas(data):
    """
    #get all unique formula names
    """
    num_calcs = len(data)
    # num_calcs = 2
    formula_list = []
    for cth in np.arange(num_calcs):
        #print(data[cth]['name'])
        formula = data[cth]['workflow']
        formula_list.append(formula)
    formula_list = np.unique(formula_list)
    # len(formula_list)
    # [x for x in formula_list if 'K10' in x]
    return formula_list


def extract_energy(data, cth):
    """ parse energy value fron energy raw data """
    #print(data[cth]['state'])
    if data[cth]['state'] == 'FAILED':
        #print('FAILURE: ', data[cth]['workflow'], data[cth]['state'])
        energy = np.nan
    elif data[cth]['state'] == 'RUN_TIMEOUT':
        #print('FAILURE: ', data[cth]['workflow'], data[cth]['state'])
        energy = np.nan
    else:
        try:
            raw_energy = data[cth]['data']['energy']
        except:
            return np.nan
        if type(raw_energy) == type([]):
            energy = raw_energy[-1] #get most relaxed energy [-1] index
        else:
            energy = raw_energy
    return energy


def extract_magmom(data, cth, name):
    """ parse magnetization value fron magnetization raw data """
    if data[cth]['state'] == 'FAILED':
        return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    if data[cth]['state'] == 'RUN_TIMEOUT':
        return [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]
    try:
        raw_magmom = data[cth]['data']['magnetization']
    except:
        return np.nan, np.nan
    if name == 'initial':
        magmom = raw_magmom[-1]
        #print(magmom[2])
        magmom_x = magmom[2]
        magmom_sites_tot = [x[-1] for x in magmom_x]
        magmom_unitcell = np.sum(magmom_sites_tot)
    elif name == 'spin':
        magmom = raw_magmom[-1]
        magmom_x = magmom[2]
        magmom_sites_tot = [x[-1] for x in magmom_x]
        magmom_unitcell = np.sum(magmom_sites_tot)
    elif name == 'afm':
        magmom = raw_magmom[-1]
        magmom_x = magmom[2]
        magmom_sites_tot = [x[-1] for x in magmom_x]
        magmom_unitcell = np.sum(magmom_sites_tot)
    elif name == 'spin_so':
        magmom = raw_magmom[-3:]
        magmom_x = magmom[0][2]
        magmom_y = magmom[1][2]
        magmom_z = magmom[2][2]
        magmom_x_sites_tot = [x[-1] for x in magmom_x]
        magmom_y_sites_tot = [x[-1] for x in magmom_y]
        magmom_z_sites_tot = [x[-1] for x in magmom_z]
        magmom_x_unitcell = np.sum(magmom_x_sites_tot)
        magmom_y_unitcell = np.sum(magmom_y_sites_tot)
        magmom_z_unitcell = np.sum(magmom_z_sites_tot)
        magmom_sites_tot = [magmom_x_sites_tot, magmom_y_sites_tot, magmom_z_sites_tot]
        magmom_unitcell = [magmom_x_unitcell, magmom_y_unitcell, magmom_z_unitcell]
    elif name == 'afm_so':
        magmom = raw_magmom[-3:]
        #print(magmom)
        magmom_x = magmom[0][2]
        magmom_y = magmom[1][2]
        magmom_z = magmom[2][2]
        magmom_x_sites_tot = [x[-1] for x in magmom_x]
        magmom_y_sites_tot = [x[-1] for x in magmom_y]
        magmom_z_sites_tot = [x[-1] for x in magmom_z]
        magmom_x_unitcell = np.sum(magmom_x_sites_tot)
        magmom_y_unitcell = np.sum(magmom_y_sites_tot)
        magmom_z_unitcell = np.sum(magmom_z_sites_tot)
        magmom_sites_tot = [magmom_x_sites_tot, magmom_y_sites_tot, magmom_z_sites_tot]
        magmom_unitcell = [magmom_x_unitcell, magmom_y_unitcell, magmom_z_unitcell]
    return magmom_sites_tot, magmom_unitcell




def get_df_optima(df_master):
    """
        Create df with min E configurations
        saved to df_gen.py
        - modified 12.5.2018 to no longer accept values wtihout spin orbit coupling
        - modified 12.19.21 to have net magnetimc moment and not only along z direction...
    """
    spinlabel = ['spin_so','afm']
    minE = []
    optMag_z = []
    optMag_x = []
    optMag_y = []
    spinlabels = []
    spindex = []
    #print(df_master.columns)
    delta_FM_AFM = []
    for c in df_master[:].iterrows():
        #print(c)
        rownum = c[0]
        try:
            energies = (c[1][['spin_so_energy','afm_energy']])
        except:
            energies = np.nan
        #print(energies)
        fm_energies = c[1]['spin_so_energy']
        afm_energies = c[1]['afm_energy']
        D_fm_afm = fm_energies - afm_energies
        magmoms = c[1][['spin_so_mag','afm_so_mag']]
        #print("magmoms==", magmoms)
        try:
            magmoms_z = [np.abs(x[2]) for x in magmoms] #take z component and absolute value
            magmoms_z = np.asarray(magmoms_z)
            magmoms_x = [np.abs(x[0]) for x in magmoms] #take z component and absolute value
            magmoms_x = np.asarray(magmoms_x)
            magmoms_y = [np.abs(x[1]) for x in magmoms] #take z component and absolute value
            magmoms_y = np.asarray(magmoms_y)
            #print("magmoms f==", magmoms)
            #magmoms = np.abs(magmoms)
            #print("2fish", energies,"MINIM", np.min(energies))
            #print("np.argwhere(ies))", np.where(energies == np.min(energies)))
            mindex_energy = np.where(energies == np.min(energies))[0][0]
            #print("mindex_energy",mindex_energy)
            optimum_mag_z = magmoms_z[mindex_energy]
            optimum_mag_x = magmoms_x[mindex_energy]
            optimum_mag_y = magmoms_y[mindex_energy]
            #print("optimum_mag=====", optimum_mag)
            optMag_z.append(optimum_mag_z)
            optMag_x.append(optimum_mag_x)
            optMag_y.append(optimum_mag_y)
            minE.append(np.min(energies))
            spindex.append(mindex_energy)
            spinlabels.append(spinlabel[mindex_energy])
            delta_FM_AFM.append(D_fm_afm)
        except:
            magmoms = np.nan
            mindex_energy = np.nan
            optimum_mag = np.nan
            optMag_z.append(np.nan)
            optMag_x.append(np.nan)
            optMag_y.append(np.nan)
            minE.append(np.nan)
            spindex.append(np.nan)
            spinlabels.append(np.nan)
            delta_FM_AFM.append(np.nan)
    df_optima = pd.DataFrame()
    df_optima['formula'] = df_master['formula'].values
    df_optima['cohesive'] = minE #added 12.23.2018  #cant rename 'energy'
    df_optima['total_elem_energy'] = df_master['total_elem_energy'].values
    df_optima['minE'] = minE
    df_optima['optMag_z'] = optMag_z #assigned to z direction
    df_optima['optMag_x'] = optMag_x
    df_optima['optMag_y'] = optMag_y
    df_optima['optMag'] = np.sqrt(np.square(optMag_x) + np.square(optMag_y) + np.square(optMag_z))
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    df_optima['delta_FM_AFM'] = delta_FM_AFM
    return df_optima



def extract_force(data, cth):
    """ parse forve value fron position_force raw data """
    if data[cth]['state'] == 'FAILED':
        #print('FAILURE: ', data[cth]['workflow'], data[cth]['state'])
        forces = np.nan
    elif data[cth]['state'] == 'RUN_TIMEOUT':
        #print('FAILURE: ', data[cth]['workflow'], data[cth]['state'])
        forces = np.nan
    else:
        try:
            raw_forces = data[cth]['data']['position_force']
        except:
            return np.nan
        if type(raw_forces[-1][0]) == type([]): #if is list of lists
            pos_forces = raw_forces[-1] #info now contains both position and forces at each position
            #print('pos_forces',pos_forces)
            forces = [x[-3:] for x in pos_forces] #get most relaxed energy [-1] index
        else: #if only entry for one relaxation step
            #print('RASW ',raw_forces)
            forces = [x[-3:] for x in raw_forces]
    return forces


def extract_position(data, cth):
    """ parse position value fron position_force raw data """
    if data[cth]['state'] == 'FAILED':
        pos = np.nan
    elif data[cth]['state'] == 'RUN_TIMEOUT':
        pos = np.nan
    else:
        try:
            raw_pos_forces = data[cth]['data']['position_force']
        except:
            return np.nan
        if type(raw_pos_forces[-1][0]) == type([]): #if is list of lists
            pos_forces = raw_pos_forces[-1] #info now contains both position and forces at each position
            #print('pos_forces',pos_forces)
            pos = [x[:3] for x in pos_forces] #get most relaxed energy [-1] index
        else: #if only entry for one relaxation step
            #print('RASW ',raw_forces)
            pos = [x[:3] for x in raw_pos_forces]
    return pos


def extract_dft_data(data):
    """
        Extract into lists for different magnetic configurations,
        the results of the dft relaxation: energy, magnetization, forces, etc.
    """
    config_names = ['initial','spin','afm','spin_so','afm_so']
    # config_names = ['afm_so']
    formula_list = gen_unique_formulas(data)
    formulas = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    energy_list = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    magmom_sites_list = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    magmom_list = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    forces_list = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    positions_list = {'initial':[],'spin':[],'afm':[],'spin_so':[],'afm_so':[]}
    num_calcs = len(data)
    for formula in formula_list:
        #print(formula)
        if formula != 'Ru4Cl12K10': #one error in your database that you need to skip
            #print(formula)
            for name in config_names:
                for cth in np.arange(num_calcs):
                    if formula == data[cth]['workflow']:
                        if data[cth]['name'] == name:
                            #print('energy :: \n', data[cth]['data']['energy'])
                            #print(cth,data[cth]['name'] )
                            formulas[name].append(formula)
                            energy = extract_energy(data, cth)
                            energy_list[name].append(energy)
                            magmom_sites, magmom_unitcell = extract_magmom(data, cth, name)
                            magmom_sites_list[name].append(magmom_sites)
                            magmom_list[name].append(magmom_unitcell)
                            forces = extract_force(data, cth)
                            forces_list[name].append(forces)
                            #positions:
                            positions = extract_position(data, cth)
                            positions_list[name].append(positions)
    return formulas, energy_list, magmom_list, magmom_sites_list, forces_list, positions_list


def generate_mag_xyz(df_magmom, descriptor):
    """
        collects x,y,z directions from dataframe
    """
    df_magmom_update = df_magmom.copy()
    mag_x_unitcell = []
    mag_y_unitcell = []
    mag_z_unitcell = []
    mag_tot_unitcell = []
    for x in df_magmom[descriptor]:
        #print(x)
        if type(x) != type([]):
            mag_x_unitcell.append(np.nan)
            mag_y_unitcell.append(np.nan)
            mag_z_unitcell.append(np.nan)
            mag_tot_unitcell.append(np.nan)
        else:
            mag_x_unitcell.append(x[0])
            mag_y_unitcell.append(x[1])
            mag_z_unitcell.append(x[2])
    ###
    # HOW to automate naming??
    df_magmom_update['fm_mag_x_unitcell'] = mag_x_unitcell
    df_magmom_update['fm_mag_y_unitcell'] = mag_y_unitcell
    df_magmom_update['fm_mag_z_unitcell'] = mag_z_unitcell
    return df_magmom_update


def findElement(df, targetatom):
    """ find which formula have particular element """
    isatom_list = []
    targetatom = targetatom + '12' #need to avoid 'Fe'
    for f in df.formula:
        isatom = targetatom in f
        isatom_list.append(isatom)
        #if targetatom in f:
        #print(f)
    isatom_list = np.asarray(isatom_list)
    return isatom_list




def gen_intersite_dist(df):
    """
        #need to use optima for pos and magmom
    """
    pos_target = 'spin_so_pos'
    mag_target = 'spin_so_mag'
    tmx_formulas = df['formula']
    positions = df[pos_target]
    magmom = df[mag_target]
    A1A2values = []
    A1Xvalues = []
    A2Xvalues = []
    X1X2values = []
    X1X3values = []
    for ith in np.arange(len(df)):
        pos_ith = np.asarray(positions[ith])
        formula_ith = tmx_formulas[ith]
        if np.isnan(pos_ith).any():
            A1A2values.append(np.nan)
            A1Xvalues.append(np.nan)
            A2Xvalues.append(np.nan)
            X1X2values.append(np.nan)
            X1X3values.append(np.nan)
        else:
            A1A2 = np.linalg.norm(pos_ith[1] - pos_ith[2])
            tmx_atoms = Atoms(formula_ith, pos_ith)
            A1aX = tmx_atoms.get_distances(1,[4,10])
            A1bX = tmx_atoms.get_distances(0,[5,11])
            A1X = np.mean([A1aX,A1bX])
            A2aX = tmx_atoms.get_distances(2,[4,6,8,10,12,14])
            A2bX = tmx_atoms.get_distances(3,[5,7,9,11,13,15])
            A2X = np.mean([A2aX,A2bX])
            X1X2 = tmx_atoms.get_distances(4,[6])
            X1X3 = tmx_atoms.get_distances(4,[8])
            #
            A1A2values.append(A1A2)
            A1Xvalues.append(A1X)
            A2Xvalues.append(A2X)
            X1X2values.append(X1X2)
            X1X3values.append(X1X3)
    return A1A2values, A1Xvalues, A2Xvalues, X1X2values, X1X3values



def find_unique_elems(data):
    """
        find unique elements using pymatgent
        - fast implementation
    """
    all_elems = []
    for f in data['formula']:
        comp = mp.Composition(f)
        elems = comp.elements
        elems = [str(x) for x in elems]
        all_elems.extend(elems)
    #print(all_elems)
    unique_elems = np.unique(all_elems)
    return unique_elems



def soap_featurize(ase_atom, unique_elems):
    """
        create soap descriptor
        # Periodic systems
    """
    rcut = 9 #7 #6 #3.1 updated 10.13.2021
    nmax = 6 #6 #4 #3
    lmax = 4 #6 #4 #3
    unique_elems = list(unique_elems)
    periodic_soap = SOAP(
        species=unique_elems, #[29],
        rcut=rcut,
        nmax=nmax,
        lmax=nmax,
        sigma=1,
        periodic=True,
        average='inner',
        sparse=False
    )
    if not pd.isnull(ase_atom):
        soap_desc = periodic_soap.create(ase_atom)
    else:
        print('have nan ase_atom')
        return np.nan
    return soap_desc

def create_ref_tmx():
    """
        # initalize ase object wtih reference POSCAR
    """
    filestem = '/Users/trevorrhone/Documents/Kaxiras/2DML/ALCF_work/theta_results/RuTMX'
    # filestem = './'  #hpc
    poscar = 'vasp/POSCAR'
    ref_cri3 = ase.io.read(filename=os.path.join(filestem, poscar))
    # ref_cri3.get_chemical_formula()
    ref_positions = ref_cri3.get_positions()
    ref_symbols = ref_cri3.symbols
    ref_cell = ref_cri3.cell
    return ref_cri3


def gen_ase_tmx(ref_cri3, energy_df):
    """ generate list of ase structures using df['formula'] and ref ase from POSCAR """
    tmx_atom_set = []
    for tmx_formula in energy_df['formula'][:]:
        #print(tmx_formula)
        if tmx_formula == 'Ru4Cl12K10':
            print('error in formula', 'Ru4Cl12K10')
        elif tmx_formula == 'CrI':
            print('error in formula', 'CrI')
        else:
            #print(tmx_formula)
            # tmx_formula = "Cr2Ru2I12"
            ref_positions = ref_cri3.get_positions()
            ref_cell = ref_cri3.cell
            # ref_atoms.set_chemical_symbols(tmx_formula)
            #print('tmx_formula',tmx_formula)
            #print('ref_positions',ref_positions)
            tmx_atoms = Atoms(tmx_formula, ref_positions)
            tmx_atoms.set_cell(ref_cell)
            tmx_atom_set.append(tmx_atoms)
            # dir(atoms)
    return tmx_atom_set


def gen_tmx_soap(tmx_atom_set, unique_elems):
    """ Create list of soap descriptors: """
    tmx_soap_list = []
    for i in np.arange(len(tmx_atom_set[:])):
        # print(i, tmx_atom_i)
        tmx_atom_i = tmx_atom_set[i]
        tmx_soap = soap_featurize(tmx_atom_i, unique_elems)
        # cri3_soap = soap_featurize(cri3, unique_elems)
        tmx_soap_list.append(tmx_soap)
    return tmx_soap_list

# functions to create unlabelled data:

def A_sub(CGT_, replace_atom1,replace_atom2, atom_config):
    """ A substitute """
    CGT = CGT_.copy()
    A1_chain = [0,3]; A2_chain = [1,2]
    A1_alt = [0,1]; A2_alt = [2,3]
    if atom_config:
        A1_config = A1_alt
        A2_config = A2_alt
    else:
        A1_config = A1_chain
        A2_config = A2_chain
    for ath in np.arange(len(A1_config)):
        site_num = A1_config[ath]
        CGT.replace(site_num, replace_atom1)
    for ath in np.arange(len(A1_config)):
        site_num = A2_config[ath]
        CGT.replace(site_num, replace_atom2)
    return CGT


def X_sub(CGT_,replace_atom_X1,replace_atom_X2):
    """ X sub """
    CGT = CGT_.copy()
    X_above_sites = [4,5,6,7,8,9]
    X_below_sites = [10,11,12,13,14,15]
    # replace_atom_X1 = 'H'
    # replace_atom_X2 = 'U'
    for site_num in X_below_sites:
        CGT.replace(site_num, replace_atom_X1)
    for site_num in X_above_sites:
        CGT.replace(site_num, replace_atom_X2)
    return CGT


def alloy_data_nextgen(CGT, Alist1, Alist2, Xlist1, Xlist2):
    """  combine A ans X site replacement for list of elements """
    abx_alloys = []
    for a1th, A1 in enumerate(Alist1):
        for a2th, A2 in enumerate(Alist2):
            for x1th, X1 in enumerate(Xlist1):
                for x2th, X2 in enumerate(Xlist2):
                    if a2th < a1th:
                        if x1th <= x2th:
                            #print(A1, A2, X1, X2)
                            replace_atom_A1 = A1
                            replace_atom_A2 = A2
                            replace_atom_X1 = X1
                            replace_atom_X2 = X2
                            #
                            # SET CHAIN OR ALTERNATING
                            alt_config = True
                            CGT_ = A_sub(CGT, replace_atom_A1, replace_atom_A2, alt_config)
                            CGT_final = X_sub(CGT_,replace_atom_X1,replace_atom_X2)
                            abx_alloys.append(CGT_final)
                            #CGT_formula = CGT.formula
                            #CGT_formula = CGT_formula.replace(' ','_')
    return abx_alloys


def load_AATMX_data(main_dir):
    """ load data from Theta """
    # data = pickle.load( open( "AATMX_data_dump.pkl", "rb" ) )
    #with open('AATMX_data_dump_mar31_2021.pkl', 'rb') as f:
    #    data = pickle.load(f)
    # with open('AATMX_data_dump_may_23_2021.pkl', 'rb') as f:
    #         data = pickle.load(f)
    with open('AAX_bilayer_data_dump_12222021.pkl', 'rb') as f:
            data = pickle.load(f)
    print("loaded data")
    formula_list = gen_unique_formulas(data)
    formula_list, energy_list, magmom_list, magmom_sites_list, forces_list, positions_list = extract_dft_data(data[:])
    print("created formula list, etc.")
    df_formula = pd.DataFrame(formula_list['initial'],columns=['formula'])
    df_energy = pd.DataFrame(energy_list)
    df_energy = df_energy.rename(columns={'initial':'initial_energy','spin':'spin_energy',
                                          'afm':'afm_energy','spin_so':'spin_so_energy','afm_so':'afm_so_energy'})
    df_magmom = pd.DataFrame(magmom_list)
    df_magmom = df_magmom.rename(columns={'initial':'initial_mag','spin':'spin_mag',
                                          'afm':'afm_mag','spin_so':'spin_so_mag','afm_so':'afm_so_mag'})

    df_forces = pd.DataFrame(forces_list)
    df_forces = df_forces.rename(columns={'initial':'initial_forces','spin':'spin_forces',
                                          'afm':'afm_forces','spin_so':'spin_so_forces','afm_so':'afm_so_forces'})
    df_pos = pd.DataFrame(positions_list)
    df_pos = df_pos.rename(columns={'initial':'initial_pos','spin':'spin_pos',
                                          'afm':'afm_pos','spin_so':'spin_so_pos','afm_so':'afm_so_pos'})

    df = pd.concat((df_formula, df_energy, df_magmom, df_forces, df_pos), axis=1)
    descriptor = 'spin_so_mag'
    df = generate_mag_xyz(df, descriptor)
    # remove F
    F_location = findElement(df, 'F')
    df = df[~F_location]
    #remove spurious data:
    df = df[df.formula != 'Ru4Cl12K10']
    df = df[df.formula != 'RuCl3_bench']
    df = df[df.formula != np.nan]
    df = df[~pd.isnull(df['formula'])]
    if not 'level_0' in df.columns:
        df = df.reset_index()
    A1A2values, A1Xvalues, A2Xvalues, X1X2values, X1X3values = gen_intersite_dist(df[:])
    df['A1A2'] = A1A2values
    df['A1X'] = A1Xvalues
    df['A2X'] = A2Xvalues
    df['X1X2'] = X1X2values
    df['X1X3'] = X1X3values
    df['energy']  = df['spin_energy'] #create summy column so can work with gen_cohesive()
    df_elements = get_unique_elem_info(main_dir, df[:], recalculate=True)
    df = gen_cohesive(df, df_elements)
    df = df[df['formula'] != 'CrI']
    df = df.reset_index(drop=True)
    return df


def gen_ax3_alloys_unlabelled(df_, save_to_disk = True):
    """
    creates ax3 alloys
    df_ : dataframe from DFT calculations
    dfsub : data from candidates
    df : dft data plus CANDIDATES
    - creates unlabelled data only, and shuffles the results (needed to avoid bias)
    - saves the result to disk
    """
    df = df_.copy(deep=True)
    ref_struct = create_ref_tmx()
    CX3_monolayer = AseAtomsAdaptor.get_structure(ref_struct)
    AX3 = CX3_monolayer.copy()
    Alist1 = ['Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd',#'Y', #no 'Y' before
              'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
              'Pr','Nd','Sm'] #stopped befpre Eu before
              #'La','Ce','Pr','Nd','Sm','Eu','Gd','Dy','Er','Yb'] #stopped befpre Eu before
              # Using bigger spacew will need bigger SOAP and lost of memery!!!
    Alist2 = ['Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Y',
              'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg',
              'Pr','Nd','Sm','Eu'] #stopped before Gd before
              #'La','Ce','Pr','Nd','Sm','Eu','Gd','Dy','Er','Yb'] #stopped before Gd before
    Xlist1 = ['Cl','Br','I','F']
    Xlist2 = ['Cl','Br','I','F']
    ax3_alloys = alloy_data_nextgen(AX3, Alist1, Alist2, Xlist1, Xlist2)
    formulas = [] #candidates
    for i in np.arange(len(ax3_alloys)):
        formula_ith = ax3_alloys[i].formula
        formula_ith = formula_ith.replace(' ','')
        formulas.append(formula_ith)
    print("TOTAL CANDIDATES: ", len(formulas))
    current_list = df['formula'].values
    mp_current_list = [mp.Composition(x) for x in current_list]
    print("len(mp_current_list) - with labels", len(mp_current_list))
    # candidates:
    mp_formulas = [mp.Composition(x) for x in formulas]
    print("len(mp_formulas) - candidates, no labels", len(mp_formulas))
    # screen formules. do not repeat those in existing df['formula']
    # check string or pymatgen object? --> Checking strings behaves strangely
    mp_unique_formulas = []
    for f in mp_formulas:
        if f not in mp_current_list[:]:
            #print(f.formula.replace(' ',''))
            mp_unique_formulas.append(f)
    dfadd = pd.DataFrame()
    unique_formulas = [x.formula.replace(' ','') for x in mp_unique_formulas]
    dfadd['formula'] = unique_formulas
    print("len(unique_formulas)", len(unique_formulas))
    # shuffle data
    dex = np.arange(len(dfadd))
    np.random.shuffle(dex)
    #print(dex[:10])
    dfadd = dfadd.iloc[dex,:]
    dfadd = dfadd.reset_index(drop=True)
    if save_to_disk:
        unlabelled_data_path = "./unlabelled_data.p"
        save_data = [dfadd, ref_struct]
        pickle.dump( save_data, open( unlabelled_data_path, "wb" ) )
    return dfadd, ref_struct


def sample_ax3_alloys(df_, unlabel_size=10, maxsize = True, initialize = False):
    """
    loads ax3 alloys from disk
    df_ : dataframe from DFT calculations
    dfsub : data from candidates
    df : dft data plus CANDIDATES
    - use this instead of gen_ax3_alloys()
    """
    df = df_.copy(deep=True)
    if initialize:
        dfadd, ref_struct = gen_ax3_alloys_unlabelled(df_, save_to_disk = True)
    # load shuffled unlabelled data from disk...
    unlabelled_data_path = "./unlabelled_data.p"
    unload_data = pickle.load( open( unlabelled_data_path, "rb" ) )
    #print(len(unload_data), unload_data)
    dfadd, ref_struct = unload_data
    if maxsize == False:
        #print('take subset')
        print("unlabel_size", unlabel_size, "dfadd shape", dfadd.shape)
        dfsub = dfadd.loc[:unlabel_size,:]
        #print('dfsub.shape',dfsub.shape)
        df = df.append(dfsub)
        #print('new df', df.shape)
    else:
        print("maxsize true", dfadd.shape, df.shape)
        df = df.append(dfadd)
        dfsub = dfadd
    df = df.reset_index(drop=True)
    return df_, dfsub, df, ref_struct


def gen_ax3_alloys(df_, unlabel_size=10, maxsize = True):
    """
    creates ax3 alloys
    df_ : dataframe from DFT calculations
    dfsub : data from candidates
    df : dft data plus CANDIDATES
    """
    df = df_.copy(deep=True)
    ref_struct = create_ref_tmx()
    CX3_monolayer = AseAtomsAdaptor.get_structure(ref_struct)
    AX3 = CX3_monolayer.copy()
    Alist1 = ['Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Y', #no 'Y' before
              'Hf','Ta','W','Re','Os','Ir','Pt','Au',
              'Pr','Nd','Sm','Eu','Gd','Er','Yb'] #stopped befpre Eu before
    Alist2 = ['Sc','Ti', 'V','Cr','Mn','Fe','Co','Ni','Cu','Zn',
              'Zr','Nb','Mo','Tc','Ru','Rh','Pd','Ag','Cd','Y',
              'Hf','Ta','W','Re','Os','Ir','Pt','Au',
              'Pr','Nd','Sm','Eu','Gd','Er','Yb'] #stopped before Gd before
    Xlist1 = ['Cl','Br','I','F']
    Xlist2 = ['Cl','Br','I','F']
    ax3_alloys = alloy_data_nextgen(AX3, Alist1, Alist2, Xlist1, Xlist2)
    formulas = [] #candidates
    for i in np.arange(len(ax3_alloys)):
        formula_ith = ax3_alloys[i].formula
        formula_ith = formula_ith.replace(' ','')
        formulas.append(formula_ith)
    print("TOTAL CANDIDATES: ", len(formulas))
    current_list = df['formula'].values
    mp_current_list = [mp.Composition(x) for x in current_list]
    print("len(mp_current_list) - with labels", len(mp_current_list))
    # candidates:
    mp_formulas = [mp.Composition(x) for x in formulas]
    print("len(mp_formulas) - candidates, no labels", len(mp_formulas))
    # screen formules. do not repeat those in existing df['formula']
    # check string or pymatgen object? --> Checking strings behaves strangely
    mp_unique_formulas = []
    for f in mp_formulas:
        if f not in mp_current_list[:]:
            #print(f.formula.replace(' ',''))
            mp_unique_formulas.append(f)
    dfadd = pd.DataFrame()
    unique_formulas = [x.formula.replace(' ','') for x in mp_unique_formulas]
    dfadd['formula'] = unique_formulas
    print("len(unique_formulas)", len(unique_formulas))
    if maxsize == False:
        #print('take subset')
        print("unlabel_size", unlabel_size, "dfadd shape", dfadd.shape)
        dfsub = dfadd.loc[:unlabel_size,:]
        #print('dfsub.shape',dfsub.shape)
        df = df.append(dfsub)
        #print('new df', df.shape)
    else:
        print("maxsize true", dfadd.shape, df.shape)
        df = df.append(dfadd)
        dfsub = dfadd
    df = df.reset_index(drop=True)
    return df_, dfsub, df, ref_struct


def load_data(df_, ref_struct, magmom = False):
    """ create data set for ML """
    print("load data df.shape", df_.shape)
    df_soap = df_.copy(deep=True)
    # update unique_elems
    unique_elems = find_unique_elems(df_)
    tmx_atom_set = gen_ase_tmx(ref_struct, df_)
    # Check to ensure got something you want
    # ase.io.write('POSCAR',ref_atoms,label=tmx_formula,direct=True, vasp5=True)
    # print(len(tmx_atom_set))
    tmx_soap_list = gen_tmx_soap(tmx_atom_set, unique_elems)
    print('len(tmx_soap_list)', len(tmx_soap_list))
    print(df_.shape)
    tmx_soap_list[0].shape
    df_soap['soap_desc'] = [x for x in tmx_soap_list]
    X = df_soap['soap_desc']
    loadresult = False
    stem = os.getcwd()
    # pull atomic descriptors
    print("Start S_tot calculations - - - - - - - - -")
    elec_prop_list = gen_P_electronic_cmpd_list(df_soap, loadresult, stem)
    Stot_list = elec_prop_list[-1]
    #print("Stot_list",Stot_list)
    #calculate J from Ising_Hamiltiaon
    deltaE = df_soap['spin_so_energy'] - df_soap['afm_so_energy']
    #J = deltaE*(1.0/(np.power(Stot_list,2)))/6
    J = deltaE*(1.0*(np.power(Stot_list,2)))/6
    df_soap['J'] = J
    #print("J",J)
    if magmom == False:
        print("magmom", magmom)
        df_optima = get_df_optima(df_soap) #added march 4, 2022
        y = df_soap['cohesive']
    else:
        print("load y --> magmom")
        print("run get_df_optima")
        df_optima = get_df_optima(df_soap)
        df_soap['optMag'] = df_optima["optMag"]
        #y = df['optMag']
        y = df_soap[['optMag','cohesive','J']]
    remove_null = False
    if remove_null:
        nulldex = ~pd.isnull(y)
        X = X[nulldex]
        y = y[nulldex]
        y = y.reset_index(drop=True)
    return X, y, df_soap, df_optima, deltaE, J, Stot_list


def load_data_TEST(df_, dfTEST, ref_struct, magmom = False):
    """
        create data set for ML
        - created version fo load_data that takes TEST set data...
        - takes in original dataframe: "df_" and dfTEST
    """
    print("load data dfTEST.shape", dfTEST.shape)
    df_soap = dfTEST.copy(deep=True)
    unique_elems = find_unique_elems(df_) #create unique_elems from original data set not TEST set
    tmx_atom_set = gen_ase_tmx(ref_struct, dfTEST)
    tmx_soap_list = gen_tmx_soap(tmx_atom_set, unique_elems)
    print('len(tmx_soap_list)', len(tmx_soap_list))
    df_soap['soap_desc'] = [x for x in tmx_soap_list]
    X = df_soap['soap_desc']
    loadresult = False
    stem = os.getcwd()
    # pull atomic descriptors
    print("Start S_tot calculations TEST---------")
    elec_prop_list = gen_P_electronic_cmpd_list(df_soap, loadresult, stem)
    Stot_list = elec_prop_list[-1]
    #print("Stot_list",Stot_list)
    #calculate J from Ising_Hamiltiaon
    deltaE = df_soap['spin_so_energy'] - df_soap['afm_so_energy']
    #J = deltaE*(1.0/(np.power(Stot_list,2)))/6
    J = deltaE*(1.0*(np.power(Stot_list,2)))/6
    df_soap['J'] = J
    #print("J",J)
    if magmom == False:
        print("magmom", magmom)
        df_optima = get_df_optima(df_soap) #added march 4, 2022
        y = df_soap['cohesive']
    else:
        print("load y --> magmom")
        print("run get_df_optima")
        df_optima = get_df_optima(df_soap)
        df_soap['optMag'] = df_optima["optMag"]
        y = df_soap[['optMag','cohesive','J']]
    return X, y, df_soap, df_optima, deltaE, J



def process_data(X,y, pca_dim=100):
    from sklearn import preprocessing
    import numpy as np
    from sklearn.decomposition import PCA
    x = np.asarray(X)  #create an array for every row of the dataset
    print("process data: x.shape, X.shape", x.shape, X.shape)
    xlist = [xx[0] for xx in x]
    xarr = np.asarray(xlist)
    x = preprocessing.scale(xarr)
    print(X.shape)
    pca = PCA(n_components=pca_dim) #300 #800 sucks
    #x = x.reshape(-1,1)
    pca.fit(x)
    X_pca = pca.transform(x)
    use_pca = True
    # create train and test data
    y = np.asarray(y)
    rand_state = None
    if use_pca:
        print("use pca")
        x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=rand_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rand_state)
    #print('x_train',x_train.shape)
    #print('x_test',x_test.shape)
    #y_train_true = y_train.copy()
    return X_pca, x, x_train, x_test, y_train, y_test, pca

def modif_train_test_split(df, X, y, unlabel_size, testsplit_ratio):
    """
    # create test_train split that keeps test set constant as add omre data to training set
    # input: df, unlabel_size, testsplit_ratio
    """
    np.random.seed(0)
    Norig = df.shape[0]
    testsplit = np.int(np.floor(testsplit_ratio*Norig))
    print("testsplit", testsplit)
    print("Norig",Norig)
    dex = np.arange(Norig)
    randex = dex.copy()
    np.random.shuffle(randex)
    testdex = randex[:testsplit]
    traindex_orig = randex[testsplit:] #original only, traindex will expand with additional unlabelled
    #will include entire set if you append unlabeldex to randex
    print("Norig, Norig+unlabel_size", Norig, Norig+unlabel_size)
    unlabel_min = Norig
    unlabel_max = Norig+unlabel_size
    if unlabel_max > X.shape[0]:
        unlabel_max = X.shape[0]
    unlabeldex = np.arange(unlabel_min, unlabel_max)
    print("Norig, Norig+unlabel_size", unlabel_min, unlabel_max)
    rand_unlabeldex = np.concatenate((randex, unlabeldex),axis=0)
    traindex = rand_unlabeldex[testsplit:]
    #print("testsplot", testsplit)
    #print(dex, randex, testdex, traindex_orig)
    #print("traindex", traindex)
    #print("unlabeldex",unlabeldex)
    #print("rand_unlabeldex",rand_unlabeldex)
    print("X.shape, traindex,testdex ", X.shape, len(traindex), len(testdex))
    x_train = X[traindex,:]
    x_test = X[testdex,:]
    y_train = y[traindex]
    y_test = y[testdex]
    return x_train, x_test, y_train, y_test


def process_data_fixed_test_set(df, X,y, unlabel_size, pca_dim=100):
    """
    fixed the size of the test set and vary amount of unlabelled data
    """
    from sklearn import preprocessing
    import numpy as np
    from sklearn.decomposition import PCA
    x = np.asarray(X)  #create an array for every row of the dataset
    xlist = [xx[0] for xx in x]
    xarr = np.asarray(xlist)
    x = preprocessing.scale(xarr)
    print(X.shape)
    pca = PCA(n_components=pca_dim) #300 #800 sucks
    #x = x.reshape(-1,1)
    pca.fit(x)
    X_pca = pca.transform(x)
    use_pca = True
    # create train and test data
    y = np.asarray(y)
    rand_state = None
    if use_pca:
        print("use pca")
        test_size=0.33
        #x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=rand_state)
        x_train, x_test, y_train, y_test = modif_train_test_split(df,X_pca,y, unlabel_size, test_size)
    else:
        #x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rand_state)
        x_train, x_test, y_train, y_test = modif_train_test_split(df,x,y, unlabel_size, test_size)
    return X_pca, x, x_train, x_test, y_train, y_test

def process_tf_data(Xy, pca_dim=100):
    from sklearn import preprocessing
    import numpy as np
    from sklearn.decomposition import PCA
    x = np.asarray(X)  #create an array for every row of the dataset
    xlist = [xx[0] for xx in x]
    xarr = np.asarray(xlist)
    x = preprocessing.scale(xarr)
    print(X.shape)
    pca = PCA(n_components=pca_dim) #300 #800 sucks
    #x = x.reshape(-1,1)
    pca.fit(x)
    X_pca = pca.transform(x)
    use_pca = True
    # create train and test data
    y = np.asarray(y)
    rand_state = None
    if use_pca:
        print("use pca")
        x_train, x_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.33, random_state=rand_state)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rand_state)
    #print('x_train',x_train.shape)
    #print('x_test',x_test.shape)
    #y_train_true = y_train.copy()
    return X_pca, x, x_train, x_test, y_train, y_test


BUFFER_SIZE = 100000

def make_ds(X, y):
    X = np.asarray(X)
    XX = np.stack([x[0] for x in X])
    y = np.asarray(y)
    ds = tf.data.Dataset.from_tensor_slices((XX, y))
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

def sort_labels_rebalance(X,y):
    """
    - create subset for unlabelled and labelled data
    - create tf.Datasets
    """
    label_dex = ~pd.isnull(y); label_dex = np.asarray(label_dex) #Exists label
    unlabel_dex = pd.isnull(y); unlabel_dex = np.asarray(unlabel_dex) #No exist labels
    pos_features = X[label_dex]
    pos_labels = y[label_dex]
    neg_features = X[unlabel_dex]
    neg_labels = y[unlabel_dex]
    ds_pos = make_ds(pos_features, pos_labels)
    ds_neg = make_ds(neg_features, neg_labels)
    balanced_dataset = tf.data.experimental.sample_from_datasets([ds_pos, ds_neg], weights=[0.5, 0.5])
    return balanced_dataset


def load_model_data(use_magmom, pca_dim = 100):
    """
    loads model data
    df -> dataset from DFT calculations
    """
    main_dir = '/Users/trevorrhone/Documents-LOCAL/RPI/Projects/AE_NN'
    print("load_AATMX_data")
    df = load_AATMX_data(main_dir)
    # reserve 10% for TEST set
    totdex = np.arange(len(df))
    np.random.shuffle(totdex)
    TESTsplit = 0.1
    testmark = np.int(np.floor(len(df)*TESTsplit))
    testdex = totdex[:testmark]
    trainvaldex = totdex[testmark:]
    dfTEST = df.iloc[testdex,:]
    dfTEST = dfTEST.reset_index(drop=True)
    df_train_val = df.iloc[trainvaldex,:]
    df_train_val = df_train_val.reset_index(drop=True)
    print("df.shape, dfTEST.shape, df_train_val.shape", df.shape, dfTEST.shape, df_train_val.shape)
    print("df from load_AATMX_Data", df.shape)
    # vary size of df_mix / df_add
    print('df shape', df.shape)
    print("run: gen_ax3_alloys")
    # df_train_val, dfadd, df_mix, ref_struct = gen_ax3_alloys(df_train_val)
    initialize = False
    df_train_val, dfadd, df_mix, ref_struct = sample_ax3_alloys(df_train_val, initialize) #4.29.2022
    print("dfadd.shape", dfadd.shape)
    print("df.shape after gen_ax3_alloys", df_train_val.shape)
    print("df_mix columns", df_mix.columns)
    print("run: load_data")
    #
    df_mix = df_mix.reset_index(drop=True) #add this just in case incex messed up #mar 9, 2022
    #
    X, y, df_tot, df_optima, deltaE, J, Stot_list = load_data(df_mix, ref_struct, use_magmom)
    XTEST, yTEST, dfTEST, df_optimaTEST, deltaE_TEST, J_TEST = load_data_TEST(df_mix, dfTEST, ref_struct, use_magmom)
    print("df.shape after load_data", df.shape)
    print("df_tot.shape after load_data", df_tot.shape)
    print("X.shape, XTEST.shape",X.shape, XTEST.shape)
    ## SHUFFLE DATA to try to uniformly distrbiute labels/unlabels
    #shuffle_dex = np.arange(len(y))
    #np.random.shuffle(shuffle_dex)
    #X = X[shuffle_dex,:]
    #y = y[shuffle_dex]
    X_pca, x, x_train, x_val, y_train, y_val, pca = process_data(X, y, pca_dim)
    maginfo =  [deltaE, J]
    #Process XTEST data:
    xtest = np.asarray(XTEST)  #create an array for every row of the dataset
    xlist = [xx[0] for xx in xtest]
    xarr = np.asarray(xlist)
    xtest = preprocessing.scale(xarr)
    print("XTEST.shape, len(xlist), xlist[0].shape, xarr.shape, xtest.shape")
    print(XTEST.shape, len(xlist), xlist[0].shape, xarr.shape, xtest.shape)
    XTEST_pca = pca.transform(xtest)
    TESTinfo = [XTEST, yTEST, dfTEST, df_optimaTEST, deltaE_TEST, J_TEST, XTEST_pca]
    return X, y, X_pca, x, x_train, x_val, y_train, y_val, df_train_val, dfadd, df_tot, df_optima, maginfo, TESTinfo, Stot_list




def load_model_data_rebalance(use_magmom, pca_dim = 100):
    """ loads model data """
    main_dir = '/Users/trevorrhone/Documents-LOCAL/RPI/Projects/AE_NN'
    print("load_AATMX_data")
    df = load_AATMX_data(main_dir)
    # vary size of df_mix / df_add
    print('df shape', df.shape)
    print("run: gen_ax3_alloys")
    #df, dfadd, df_mix, ref_struct = gen_ax3_alloys(df)
    df, dfadd, df_mix, ref_struc = sample_ax3_alloys(df) #4.29.2022
    print("dfadd.shape", dfadd.shape)
    print("run: load_data")
    X, y, df, df_optima, deltaE, J, Stot_list = load_data(df_mix, ref_struct, use_magmom)
    # balance:
    balanced_dataset = sort_labels_rebalance(X,y)
    # how to split up again into X, Y for process_data???
    X_pca, x, x_train, x_test, y_train, y_test, pca = process_data(X, y, pca_dim)
    return X, y, X_pca, x, x_train, x_test, y_train, y_test, df, dfadd, df_mix, df_optima, [deltaE, J]


def Xy_gen(pca_dim, use_magmom, reload_data):
    """
    load X, y from recalculated data or saved data
    input: pca_dim, use_magmom, reload_data
    """
    load_data_path = f'load_data_output_pca{pca_dim}.p'
    if reload_data:
        X, y, X_pca, x, x_train, x_val, y_train, y_val, df, dfadd, df_mix, df_optima, maginfo, TESTinfo, Stot_list = load_model_data(use_magmom, pca_dim)
        load_data_output = [X, y, X_pca, x, x_train, x_val, y_train, y_val, df, dfadd, df_mix, df_optima, maginfo, TESTinfo]
        pickle.dump( load_data_output, open( load_data_path, "wb" ) )
        [deltaE, J, Stot_list] = maginfo
        [XTEST, yTEST, dfTEST, df_optimaTEST, deltaE_TEST, J_TEST, XTEST_pca] = TESTinfo
    else:
        load_data_output = pickle.load( open( load_data_path, "rb" ) )
        [X, y, X_pca, x, x_train, x_val, y_train, y_val, df, dfadd, df_mix, df_optima, maginfo, TESTinfo] = load_data_output
        [deltaE, J, Stot_list] = maginfo
        [XTEST, yTEST, dfTEST, df_optimaTEST, deltaE_TEST, J_TEST, XTEST_pca] = TESTinfo
    return X, y, X_pca, x, x_train, x_val, y_train, y_val, df, dfadd, df_mix, df_optima, maginfo, TESTinfo



if __name__ == "__main__":
    #pca_dim = 200
    load_model_data()
