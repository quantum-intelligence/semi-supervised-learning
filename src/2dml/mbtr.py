# py file for functions written is MBTR_catalysis
# For use with catalysis.ipynb file

# COPIED TO MBTR.PY SEPT21 6:20pm

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
import pickle
import os.path
import seaborn as sns
from ase.spacegroup import Spacegroup
from math import floor, ceil
import ase.db
from matplotlib import cm
import math
import operator
from mbtr_functions import *
from pymatgen import Lattice, Structure, Molecule
import re
import pymatgen as mp

# NOTES:
# - consider i=j case separatelu
# - account for double counting of entires of off-diagonal terms





def get_smilesbobs(df_smiles,sigma,min_dist,on_the_fly):
    """
        calculates smiles bobs from df_smiles
    """
    unique_names = df_smiles['cids'][~pd.isnull(df_smiles['cids'])].values
    #unique_namesnames = unique_names[:]
    unique_namesnames = unique_names[:]
    unique_names = (unique_names).astype(int)
    #unique_names_met = np.append(297,unique_names) #297 is methane cid
    #unique_names_met = unique_names_met.tolist()
    unique_names = unique_names.tolist()
    #             get_BoBvectors_debug(molecule_list, species, identifier, min_dist, on_the_fly):
    smilesvectors = get_BoBvector_cids(unique_names,'cid',min_dist,on_the_fly,df_smiles)
    smilesbobs = get_Bobfeatures(smilesvectors,sigma)
    return smilesbobs





def mol_matrix(molecule_name,identifier,on_the_fly,df_smiles):
    """
        returms an adjacency matrix, mol_matrix, that represents a molecule
        identifier can be 'name' or 'cids'-->  design up accepts CIDS
        updated: to get [atom_bag, coords, Z ] from df_smiles.csv and
                not from pcp.get_compounds()
    """
    if type(molecule_name) == type(np.nan):
        #np.isnan(molecule_name) does NOT work
        return np.nan
    else:
        # IUOAC_NAME for methyl radical is not ready by pubchempy for some reason...
        if molecule_name == '$l^{3}-carbane':
            molecule_name = 'Methyl radical'
        if on_the_fly == True:
            molecule = pcp.get_compounds(molecule_name,identifier)[0]
            atom_bag, coords, Z = get_molcoords(molecule)
        else:
            if (df_smiles['cids'] == molecule_name ).any() == False:
                atom_bag = []
                coords = np.nan
                Z = np.nan
            else:
                atom_bag = df_smiles[(df_smiles['cids'] == molecule_name )]['atom_bag_list'].values[0]
                coords = df_smiles[(df_smiles['cids'] == molecule_name )]['coords_list'].values[0]
                coords = coords.replace('[','')
                #convert coords from string to numpy 2d-array
                coords = coords.replace(']','')
                coords = np.fromstring(coords, sep=',')#.reshape(-1, 4)
                coords = coords.reshape(-1,3)
                #
                Z = df_smiles[(df_smiles['cids'] == molecule_name )]['Z_list'].values[0]
                Z = Z.replace('[','')
                Z = Z.replace(']','')
                Z = np.fromstring(Z, sep=',')#.reshape(-1, 4)
                if type(atom_bag) == str:
                    atom_bag = str_to_list(atom_bag)

        N = len(atom_bag)
        if N > 0:
            MolMatrix = np.zeros((2,N,N),dtype=object)
            # Use pymatgen_molecue class
            pm_molecule = Molecule(atom_bag, coords)
            mol_dm = pm_molecule.distance_matrix
            bond_list = []
            for i in np.arange(N):
                atom_i = atom_bag[i]
                for j in np.arange(N):
                    if j >= i:
                        atom_j = atom_bag[j]
                        if i == j:
                            bondpair = atom_i
                            bond_list.append(bondpair)
                            MolMatrix[0,i,j] = bondpair
                            #MolMatrix[1,i,j] = 0.5*Z[i]**2.4  #/mol_dm[i,j]
                            #MolMatrix[1,i,j] = mol_dm[i,j]
                            MolMatrix[1,i,j] = 0.5*Z[i]**2.4
                        else:
                            bondpair = atom_i + atom_j
                            bond_list.append(bondpair)
                            #MolMatrix[1,i,j] = mol_dm[i,j]
                            #MolMatrix[1,i,j] = Z[i]*Z[j]
                            MolMatrix[1,i,j] = Z[i]*Z[j]/mol_dm[i,j]
                            MolMatrix[0,i,j] = bondpair
                    MolMatrix[0,j,i] = MolMatrix[0,i,j]
                    MolMatrix[1,j,i] = MolMatrix[1,i,j]
            uniquepair = np.unique(bond_list)

        else:
            MolMatrix = np.nan
    return MolMatrix, atom_bag




#copied from notebook 8/11
def make_bag(MolMatrix, min_dist):
    """
        Make a bag - collection of bondtypes and corresponding property
        input: MolMatrix and minimum interatomic distances
        output: bondtypes, bag (i.e contents of each bag) , counts (i.e. lenght of each bag)
        This function fills the bag with the property of interatomic distance
    """
    if type(MolMatrix) == type(np.nan):
        #print 'get here'
        return [], [], 0
    else:
        # make bags with MolMatrix as input:
        bondtypes = np.unique(MolMatrix[0,:,:])
        # number of bonds first
        bag = []
        for bond in bondtypes:
            #print '\n', bond, '\n'
            bond_index = np.argwhere(MolMatrix[0,:,:] == bond)
            #num_atoms = MolMatrix.shape[1]
            #num_nn = (len(np.argwhere(MolMatrix[1,:,:] <= min_dist))-num_atoms)/2.0
            #print bond_index
            bond_counts = [] #for a given bondtype get the list of matrix_Values
            for b_dex in bond_index:
                #print b_dex
                #print 'dex', b_dex[0], b_dex[1]
                mat_val = MolMatrix[1,b_dex[0],b_dex[1]]
                #print mat_val
                bond_counts.append(mat_val)
                #print bond_counts
            bag.append(bond_counts) #collect the list of matrix_values for each type of bond
        counts = [len(b) for b in bag]
    return bondtypes, bag, counts




def getMasterMatrix(bag, bond_counts, master_list, bondtypes):
    """
        Fill Master Matrix with values vased on master_list
    """
    # Create master matrix:
    if type(bag) == type(np.nan):
        return np.nan
    #max_rowsize = np.max(bond_counts) # THIS VARIES depending on molecule..
    max_rowsize = 350                  # Want to maximize this.
                                       # That is, consider for biggest molecule
    max_colsize = len(master_list)
    MasterMatrix = np.zeros((max_rowsize,max_colsize))
    for ith, bth in enumerate(master_list):
        # print bth, ith
        # bdex = in_list(bth,bondtypes)
        bdex = np.argwhere(bondtypes == bth)
        # print bdex
        if len(bdex) == 0:
            #print 'SHIOT', sizeentry, ith
            #print max_rowsize, MasterMatrix.shape
            MasterMatrix[:,ith] = np.zeros(max_rowsize)
        else:
            bdex = bdex[0][0]#-1 #count list from 1st element but array from 0th..
            bag_element = bag[bdex]
            sizeentry = len(bag_element)
            #print 'MasterMatrix.shape', MasterMatrix.shape, 'bedex', bdex
            #print 'length bag_element ', sizeentry
            MasterMatrix[:sizeentry,bdex] = bag_element
    return MasterMatrix

#bondtypes, bag, counts = make_bag(MolMatrix, min_dist)
#molecule_names = cid_to_name(df2,'Reactant 1 CID')




def get_master_set(molecule_list, min_dist,identifier,df_smiles):
    """
        Defined in mbtr.py
        Create a set of MasterMatrices from the data
    """
    master_list = get_masterlist(molecule_list, min_dist,identifier,df_smiles)
    master_matrix_set = []
    # print master_list
    for molname in molecule_list:
        MolMatrix = mol_matrix(molname,identifier)
        bondtypes, bag, bond_counts = make_bag(MolMatrix, min_dist)
        ###print bondtypes, bag, bond_counts, '\n'
        master_matrix = getMasterMatrix(bag, bond_counts, master_list, bondtypes)
        master_matrix_set.append(master_matrix)
        # print master_matrix[:,2]
    return master_matrix_set



def get_BoBvectors(df2,species,identifier,df_smiles):
    """
        Get bob_vectors
        INPUT : 'Reactant X CID'. Converts CID to names using cid_to_name()
    """
    # molecule_list = ['methane', 'ethane', 'methanol', 'ethanol', 'acetone',
    #                 'phenol','METHYLAMINE','benzene','styrene','cyclohexane']
    # molecule_list = df2[species]
    molecule_list = cid_to_name(df2,species)
    min_dist = 1.1
    master_matrix_set = get_master_set(molecule_list, min_dist,identifier,df_smiles)
    # CREATE BoB vectors from Master Matrices...
    set_size = len(master_matrix_set)
    # print set_size
    vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    BoB_vector = np.zeros((vec_length, set_size))
    for mth, mat_item in enumerate(master_matrix_set):
        #print mat_item.shape
        vec_item = np.reshape(mat_item,(1,-1), order='F')
        BoB_vector[:, mth] = vec_item
    return BoB_vector



def get_BoBvector_cids(molecule_names,identifier,df_smiles):
    """
        Get bob_vector from a list of molecule names (in a list format)
        INPUT : name of molecule, placed inside a list!!
    """
    min_dist = 1.1
    master_matrix_set = get_master_set(molecule_names, min_dist,identifier, df_smiles)
    # CREATE BoB vectors from Master Matrices...
    set_size = len(master_matrix_set)
    # print set_size
    vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    BoB_vector = np.zeros((vec_length, set_size))
    for mth, mat_item in enumerate(master_matrix_set):
        #print mat_item.shape
        vec_item = np.reshape(mat_item,(1,-1), order='F')
        BoB_vector[:, mth] = vec_item
    return BoB_vector


def str_to_list(aa):
    """
     string to list for atom_bag
    """
    xx = []
    for a in aa:
        if a.isalnum():
            if not a == 'u':
                xx.append(a)
    return xx


def get_BoBvectors_debug(molecule_list, species, identifier, min_dist, on_the_fly,df_smiles):
    """
        Get bob_vectors
        INPUT : 'Reactant X CID'. Converts CID to names using cid_to_name()
    """
    master_matrix_set, atom_bag_list = get_master_set_multi(molecule_list, min_dist,identifier,on_the_fly,df_smiles)
    set_size = len(master_matrix_set)
    #print 'set size', set_size
    #vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    vec_length = master_matrix_set[0].shape[1]*master_matrix_set[0].shape[2]
    NUM_flavors = 4
    Bob_vector = np.zeros((vec_length, set_size), dtype = np.float)
    Bob_vector_flavors = np.empty((NUM_flavors,vec_length, set_size), dtype = np.float)

    for fth in np.arange(NUM_flavors):
        for mth, mat_item in enumerate(master_matrix_set):
            vec_item = np.reshape(mat_item[fth,:,:], (1,-1), order = 'F')
            #print 'mat item shape', mat_item.shape
            #print 'vec item shape', vec_item.shape
            Bob_vector[:, mth] = vec_item
        Bob_vector_flavors[fth,:,:] = Bob_vector
    return Bob_vector_flavors



def key_bondstructures(df2b):
    """gets pertinent bond structure moeities"""
    species_list = ['Reactant 1', 'Reactant 2','Reactant 3', 'Major product']
    for species in species_list:
        #print species
        df_prints = get_subprints(df2b,species) # GETS Bond Structure Moeities...
        #print 'got here'
        df2b = pd.concat([df2b,df_prints], axis=1, ignore_index=False)
    return df2b



def bob_update(df2, sigma, min_dist,on_the_fly,p,df_smiles,NUM_flavors):
    """
        Updates bob chemical space values for r1_bob, p1_bob, etc.
    """
    # Add sub_fingerprint info to dataframe for the species_list:
    df2b = df2.copy()
    # *****
    # Calculate results and Create pickled file if does not exists:
    # get_bob_df(df2b)
    recalculate = True
    filepath = '/Users/trevorrhone/Documents/Kaxiras/FRiend/bob_space.p'
    if not os.path.exists(filepath) or recalculate:
        print('recalculating bob_df')
        df_bob_list, rth_bob_space,atoms_list = get_bob_multi(df2b,sigma,min_dist,on_the_fly,p,df_smiles,NUM_flavors)
        print('df_bob_list', df_bob_list)
        #update df after calculating bob chem space
        for fth in np.arange(NUM_flavors):
            #first bob_space index is for reactant/prodcut.. second index is for chem space flavor
            print('df_bob_list[fth]', df_bob_list[fth])
            for rth, val in enumerate(df_bob_list[fth]):
                print('rth', rth)
                print(len(rth_bob_space[rth][fth]))
                df2b[val] = rth_bob_space[rth][fth]
            pickle.dump( rth_bob_space, open( "bob_space.p", "wb" ) )
    else:
        print('loading pickled file')
        rth_bob_space = pickle.load( open( "bob_space.p", "rb" ) )
        for fth in np.arange(NUM_flavors):
            print('df_bob_list[fth]', df_bob_list[fth])
            for rth, val in enumerate(df_bob_list[fth]):
                df2b[val] = rth_bob_space[rth][fth]

    if not u'p>1N' in df2b.columns: #sometimes columns are duplicated!!!
        print('adding key_bondstructers')
        df2b = key_bondstructures(df2b)
    else:
        print('key bondstrucutres already present')
    #key_bondstructures(df2b)
    df2 = df2b.copy() # don't concat multiple times if run this
                      # ell more than once
    return df2


def get_Bobfeatures_multi(Bob_vector_flavors,sigma,p):
    """
        # Test implementatino by definning similarity metrix and plotting results
        # SHould use sigma as a tuning parameter
        # L_p norm delimited by p
        OUTPUT: set of f flavors of chemspace values of set of molecules of one of the reactants or producnts
    """
    f_dim = Bob_vector_flavors.shape[0]
    i_dim = Bob_vector_flavors.shape[2]
    features_flavors = np.empty((f_dim,i_dim))
    NUM_flavors = Bob_vector_flavors.shape[0]
    print('NUM_flavors', NUM_flavors)
    for fth in np.arange(NUM_flavors):
        ref = np.zeros(Bob_vector_flavors[0,:,:].shape[0], dtype = np.float)
        Bob_vector = Bob_vector_flavors[fth,:,:]
        features = np.empty(i_dim)
        for ith in np.arange(len(Bob_vector[0,:])):
            delta = np.abs(ref - Bob_vector[:,ith])
            dif = np.sum(delta**p)
            kernel = 1.0*dif
            #kernel = np.exp(-1.0*dif/sigma)
            #features.append(kernel)
            features[ith] = kernel
        #features_flavors.append(features)
        features_flavors[fth,:] = features
    return features_flavors



def make_bag_multi(MolMatrix, min_dist):
    """
        Make a bag - collection of bondtypes and corresponding property
        input: MolMatrix and minimum interatomic distances
        output: bondtypes, bag (i.e contents of each bag) , counts (i.e. lenght of each bag)
        This function fills the bag with the property of interatomic distance
    """
    if type(MolMatrix) == type(np.nan):
        #print 'get here'
        return [], [[],[],[],[],[]], 0
        #return [], [[]], 0
    else:
        # make bags with MolMatrix as input:
        bondtypes = np.unique(MolMatrix[0,:,:])
        # number of bonds first
        bag = []
        bag_PE = []
        bag_IR = []
        bag_AR = []
        bag_OX = []
        for bond in bondtypes:
            #print '\n', bond, '\n'
            bond_index = np.argwhere(MolMatrix[0,:,:] == bond)
            bond_counts = [] #for a given bondtype get the list of matrix_Values
            bond_counts2 = []
            bond_counts3 = []
            bond_counts4 = []
            bond_counts5 = []
            for b_dex in bond_index:
                mat_val = MolMatrix[1,b_dex[0],b_dex[1]]
                mat_val2 = MolMatrix[2,b_dex[0],b_dex[1]]
                mat_val3 = MolMatrix[3,b_dex[0],b_dex[1]]
                mat_val4 = MolMatrix[4,b_dex[0],b_dex[1]]
                mat_val5 = MolMatrix[5,b_dex[0],b_dex[1]]
                #if MolMatrix[1,b_dex[0],b_dex[1]] <= min_dist:   #min_dist is not distance! is columb matrix value
                bond_counts.append(mat_val)
                bond_counts2.append(mat_val2)
                bond_counts3.append(mat_val3)
                bond_counts4.append(mat_val4)
                bond_counts5.append(mat_val5)
            bag.append(bond_counts) #collect the list of matrix_values for each type of bond
            bag_PE.append(bond_counts2)
            bag_IR.append(bond_counts3)
            bag_AR.append(bond_counts4)
            bag_OX.append(bond_counts5)
        counts = [len(b) for b in bag]
        bag_set = [bag, bag_PE, bag_IR, bag_AR, bag_OX]
        #bag_set = [bag]
    return bondtypes, bag_set, counts



def get_bob_multi(df2,sigma,min_dist,on_the_fly,p,df_smiles,NUM_flavors):
    """
        UPDATE DF WITH BOB FEATURE VECTORS
        NOTE:
        - First index is rth reactant/prodcut.. inner list index is flavor
        - in rth_bob_space
    """
    # bet features and Bobvectors for Reactants 2 ,3 and products et...
    #
    #reaction_list = ['Reactant 1 CID','Reactant 2 CID','Reactant 3 CID','Major Product CID']
    reaction_list = ['Reactant 1 CID','Reactant 2 CID','Reactant 3 CID','Major Product CID','Product 2 CID','Product 3 CID']
    df_bob_list = [['r1_bob','r2_bob','r3_bob','p1_bob','p2_bob','p3_bob'],
                   ['r1_PE','r2_PE','r3_PE','p1_PE','p2_PE','p3_PE'],
                   ['r1_IR','r2_IR','r3_IR','p1_IR','p2_IR','p3_PE'],
                   ['r1_AR','r2_AR','r3_AR','p1_AR','p2_AR','p3_AR'],
                   ['r1_OX','r2_OX','r3_OX','p1_OX','p2_OX','p3_OX']]
    rth_bob_space = []
    #for fth in np.arange(NUM_flavors):
    atoms_reaction_list = []
    for rth, react in enumerate(reaction_list):
        print(rth, react)
        Bob_vector_flavors, atoms_list = get_BoBvectors_multi(df2,react,'cid',min_dist,on_the_fly,df_smiles,NUM_flavors)
        features_flavors = get_Bobfeatures_multi(Bob_vector_flavors,sigma,p)
        atoms_reaction_list.append(atoms_list)
        rth_bob_space.append(features_flavors)
        # NOTE:
        # First index is rth reactant/prodcut.. inner list index is flavor
    return df_bob_list, rth_bob_space, atoms_reaction_list



def get_BoBvector_cids_multi(molecule_list,identifier,min_dist,on_the_fly,df_smiles,NUM_flavors):
    """
        Get bob_vectors
        INPUT : 'Reactant X CID'. Converts CID to names using cid_to_name()
    """
    #NUM_flavors = 5
    master_matrix_set, atom_bag_list = get_master_set_multi(molecule_list, min_dist,identifier,on_the_fly,df_smiles,NUM_flavors)
    set_size = len(master_matrix_set)
    #print 'set size', set_size
    #vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    vec_length = master_matrix_set[0].shape[1]*master_matrix_set[0].shape[2]
    #print 'mm shape', master_matrix_set[0].shape
    #print 'vec length', vec_length
    #print 'bob vector shape', Bob_vector.shape
    Bob_vector = np.zeros((vec_length, set_size), dtype = np.float)
    Bob_vector_flavors = np.empty((NUM_flavors,vec_length, set_size), dtype = np.float)
    for fth in np.arange(NUM_flavors):
        for mth, mat_item in enumerate(master_matrix_set):
            vec_item = np.reshape(mat_item[fth,:,:], (1,-1), order = 'F')
            Bob_vector[:, mth] = vec_item
        Bob_vector_flavors[fth,:,:] = Bob_vector
    return Bob_vector_flavors, atom_bag_list



def get_BoBvectors_multi(df2,species,identifier,min_dist,on_the_fly,df_smiles,NUM_flavors):
    """
        Get bob_vectors
        INPUT : 'Reactant X CID'. Converts CID to names using cid_to_name()
    """
    molecule_list = df2[species]
    #min_dist = 40.0
    #master_matrix_flavors = get_master_set_multi(molecule_list, min_dist,identifier,on_the_fly)
    master_matrix_set, atom_bag_list = get_master_set_multi(molecule_list, min_dist,identifier,on_the_fly,df_smiles,NUM_flavors)
    #for master_matrix_set in master_matrix_flavors:
    # CREATE BoB vectors from Master Matrices...
    set_size = len(master_matrix_set)
    #vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    vec_length = master_matrix_set[0].shape[1]*master_matrix_set[0].shape[2]
    #NUM_flavors = 5
    Bob_vector = np.zeros((vec_length, set_size), dtype = np.float)
    Bob_vector_flavors = np.empty((NUM_flavors,vec_length, set_size), dtype = np.float)

    for fth in np.arange(NUM_flavors):
        for mth, mat_item in enumerate(master_matrix_set):
            vec_item = np.reshape(mat_item[fth,:,:], (1,-1), order = 'F')
            Bob_vector[:, mth] = vec_item
        Bob_vector_flavors[fth,:,:] = Bob_vector
    return Bob_vector_flavors, atom_bag_list


def get_masterlist(molecule_list, min_dist,identifier,on_the_fly,df_smiles,NUM_flavors):
    """
        generates a master list of bondtypes (list of unique items) from the dataset
        identifier can be 'name' or 'cid'
    """
    bondtypes_list = []
    #bag_list = []
    # Defune molecule & adjacency matrix
    for molname in molecule_list:
        molecule_name = molname
        #print 'molecule_name', molecule_name
        MolMatrix, atom_bag = mol_matrix_multi(molecule_name,identifier,on_the_fly,df_smiles,NUM_flavors)
        #print atom_bag
        # Define bag and set of bondtypes
        bondtypes, bag, bond_counts = make_bag_multi(MolMatrix, min_dist)
        bondtypes_list.append(bondtypes)
        #bag_list.append(bag)
        # Need to be able to compare different molecules. Pad with zeros.
        # Hard code the number of zeros? or be ableto do this on the fly

    # Create MAster list of bonds:
    # from set of bond types loop through each one and collect master list
    combined_list = []
    for s in bondtypes_list:
        s = list(s)
        combined_list.append(s)
    reduced_list = reduce(operator.concat, combined_list)
    master_list = np.unique(reduced_list)
    #print 'master list', master_list
    return master_list


def get_master_set_multi(molecule_list, min_dist, identifier,on_the_fly,df_smiles,NUM_flavors):
    """
        Defined in mbtr.py
        Create a set of MasterMatrices from the data
        updates: update to only accept CIDS numbers...
    """
    master_list = get_masterlist(molecule_list, min_dist, identifier, on_the_fly,df_smiles,NUM_flavors) #molecule_list is CIDS list
    master_matrix_set = []
    # print master_list
    master_matrix_flavors = []
    atom_bag_list = []
    for molname in molecule_list:
        MolMatrix, atom_bag = mol_matrix_multi(molname,identifier,on_the_fly,df_smiles,NUM_flavors)
        bondtypes, bag_set, bond_counts = make_bag_multi(MolMatrix, min_dist)
        atom_bag_list.append(atom_bag)
        ###print bondtypes, bag, bond_counts, '\n'
        master_matrix = getMasterMatrix_multi(bag_set, bond_counts, master_list, bondtypes,NUM_flavors)
        master_matrix_set.append(master_matrix)

    return master_matrix_set, atom_bag_list


### COPIED FROM MBTR Aug16:
### UPDATED SINCE COPY!!!

def get_smilesbobs(df_smiles,sigma,min_dist,on_the_fly,p,NUM_flavors):
    """
        calculates smiles bobs from df_smiles
    """
    unique_names = df_smiles['cids'][~pd.isnull(df_smiles['cids'])].values
    #unique_namesnames = unique_names[:]
    unique_namesnames = unique_names[:]
    unique_names = (unique_names).astype(int)
    #unique_names_met = np.append(297,unique_names) #297 is methane cid
    #unique_names_met = unique_names_met.tolist()
    unique_names = unique_names.tolist()
    #             get_BoBvectors_debug(molecule_list, species, identifier, min_dist, on_the_fly):
    smilesvectors, atoms_list = get_BoBvector_cids_multi(unique_names,'cid',min_dist,on_the_fly,df_smiles,NUM_flavors)
    smilesbobs = get_Bobfeatures_multi(smilesvectors,sigma,p)
    return smilesbobs



def create_smilesbobs(smilesRECALCULATE, df_smiles,sigma, min_dist,on_the_fly,p):
    """
        creates smiles bobs from df_smiles
    """
    if smilesRECALCULATE == True:
        ## throttle_smilesbobs:
        dsize =  500
        N = len(df_smiles)
        steps = N/dsize*1.0
        numsteps = np.floor(steps)

        step_sequence = np.arange(0,numsteps*dsize,dsize)
        step_sequence = np.asarray(step_sequence, dtype = int)
        smilesbobs_list = []

        for ith, i in enumerate(step_sequence[:]):
            sub_df_smiles = df_smiles[i:i+dsize]
            sub_smilesbobs = get_smilesbobs(sub_df_smiles, sigma, min_dist,on_the_fly,p,NUM_flavors)
            smilesbobs_list.append(sub_smilesbobs)
            #time.sleep(90)
        sub_df_smiles = df_smiles[step_sequence[-1] + dsize: N]
        sub_smilesbobs = get_smilesbobs(sub_df_smiles, sigma,min_dist,on_the_fly,p,NUM_flavors)
        smilesbobs_list.append(sub_smilesbobs)

        pickle.dump( smilesbobs_list, open( "smilesbobs_list.p", "wb" ) )
        bobfileloc = r'/Users/trevorrhone/Documents/Kaxiras/FRiend/smilesbobs_saved.csv'

        #####
        for i, s in enumerate(smilesbobs_list):
            if i == 0:
                smilesconcat = s
            else:
                smilesconcat = np.concatenate((smilesconcat,s),axis=1)


        df_smilesbobs = pd.DataFrame(smilesconcat)
        #np.savetxt(bobfileloc, smilesbobs, fmt='%.18e', delimiter = ',')
        #THESE ARE NOT 'UNPACKED SMILESBOBS... EACH ROW REPRESENTS ITH STEP_SEQUENCE RESULT
        df_smilesbobs.to_csv(bobfileloc, index=True, mode='w', sep=',')
        pickle.dump( smilesconcat, open( "smilesbobs.p", "wb" ) )
        smilesbobs = smilesconcat #reset label for agreement wiht the rest of the code you've written
        return smilesbobs


def getMasterMatrix_multi(bag_set, bond_counts, master_list, bondtypes, NUM_flavors):
    """
        Fill Master Matrix with values vased on master_list
    """
    bag = bag_set[0]
    bag_PE = bag_set[1]
    bag_IR = bag_set[2]
    bag_AR = bag_set[3]
    bag_OX = bag_set[4]
    # Create master matrix:
    if type(bag) == type(np.nan):
        return np.nan
    #max_rowsize = np.max(bond_counts) # THIS VARIES depending on molecule..
    max_rowsize = 2000                  # Want to maximize this.
                                       # That is, consider for biggest molecule
    max_colsize = len(master_list)
    MasterMatrix = np.zeros((NUM_flavors, max_rowsize, max_colsize),dtype=np.float)
    for ith, bth in enumerate(master_list):
        # bdex = in_list(bth,bondtypes)
        bdex = np.argwhere(bondtypes == bth)
        if len(bdex) == 0:
            MasterMatrix[0,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[1,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[2,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[3,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[4,:,ith] = np.zeros(max_rowsize, dtype=np.float)
        else:
            bdex = bdex[0][0]#-1 #count list from 1st element but array from 0th..
            bag_element = bag[bdex]
            bag_PE_element = bag_PE[bdex]
            bag_IR_element = bag_IR[bdex]
            bag_AR_element = bag_AR[bdex]
            bag_OX_element = bag_OX[bdex]
            sizeentry = len(bag_element)
            MasterMatrix[0, :sizeentry, bdex] = bag_element
            MasterMatrix[1, :sizeentry, bdex] = bag_PE_element
            MasterMatrix[2, :sizeentry, bdex] = bag_IR_element
            MasterMatrix[3, :sizeentry, bdex] = bag_AR_element
            MasterMatrix[4, :sizeentry, bdex] = bag_OX_element
    return MasterMatrix



def make_tree_bob_space(df2,df_smiles_nna,constrain,use_weight):
    """
        location: tree_functions.py
        Use df_unique_smiles as dictionary to look up values of df2['product tree smiles']
        - allows to use wither 'possible_products' or 'constrain_prodcuts'
        - have to specify target cs type
    """
    bob_bag = []
    #
    #
    target_cs = 'smilebobs_OX'
    #
    #
    # loop through each row of df2
    if constrain == True:
        if use_weight == True:
            tree_products =  u'weight_constrain_products'
        else:
            tree_products = 'atoms_constrain'
    else:
        tree_products = 'possible_products'
    #
    for ith, smiles_bag in enumerate(df2[tree_products]):
        bob_set = []
        #loop through a list of possible products in a given row of df2 (ie a given reaction)
        if len(np.atleast_1d(smiles_bag)) == 1:
            sm_index = df_smiles_nna.index[df_smiles_nna.loc[:,'smiles'] == str(smiles_bag[0])]
            #bob = df_smiles_nna.loc[sm_index,'smilebobs_OX'].values[0]
            bob = df_smiles_nna.loc[sm_index, target_cs].values[0]
            bob_set.append(bob)
        else:
            for smile in smiles_bag:
                #sm_index = np.where(smile == df_smiles_nna['smiles'])
                sm_index = df_smiles_nna.index[df_smiles_nna.loc[:,'smiles'] == smile]
                if len(sm_index) != 0:
                    #sm_index = int(sm_index[0][0]) # TAKE FIRST ELEment if there are duplicates
                    #bob = df_smiles_nna.loc[sm_index,'smilebobs_OX'].values
                    bob = df_smiles_nna.loc[sm_index, target_cs].values
                    if len(bob) == 1:
                        bob = bob[0]
                    elif len(bob) > 1:
                        if bob[0] != bob[1]:
                            print('are duplicates', bob)
                        bob = bob[0]
                    bob_set.append(bob)
                else:
                    bob_set.append(np.nan)
        bob_bag.append(bob_set)
    return bob_bag



## UPDATED CODES:

def mol_matrix_multi(molecule_name,identifier,on_the_fly,df_smiles,NUM_flavors):
    """
        returms an adjacency matrix, mol_matrix, that represents a molecule
        identifier can be 'name' or 'cids'-->  design up accepts CIDS
        updated: to get [atom_bag, coords, Z ] from df_smiles.csv and
                not from pcp.get_compounds()
        PE - Pauli electronegativity
        IR - ionic radius
        AR - atomic radius
        - normalize by heavy atom count!!
    """
    # Return nan and empty list in null case
    #ERROR in this if statemetn with algo thinking
    #if type(molecule_name) == type(np.nan):
    #    #np.isnan(molecule_name) does NOT work
    #    return np.nan, []
    if pd.isnull(molecule_name) == True:
        return np.nan, []
    else:
        # IUOAC_NAME for methyl radical is not ready by pubchempy for some reason...
        if molecule_name == '$l^{3}-carbane':
            molecule_name = 'Methyl radical'
        if on_the_fly == True:
            molecule_name = np.int(molecule_name)
            molecule = pcp.get_compounds(molecule_name,identifier)[0]
            heavy_atom_count = molecule.heavy_atom_count
            atom_bag, coords, Z = get_molcoords(molecule)
            PE_bag = [1.0*mp.Element(i).X  for i in atom_bag]
            IR_bag = [1.0*mp.Element(i).average_ionic_radius  for i in atom_bag]
            AR_bag = [1.0*mp.Element(i).atomic_radius  for i in atom_bag]
            OX_bag = [1.0*mp.Element(i).common_oxidation_states[0]  for i in atom_bag]
        else:

            if (df_smiles['cids'] == molecule_name ).any() == False:
                #molecule_name = int(molecule_name) #get error
                molecule = pcp.get_compounds(np.int(molecule_name),identifier)[0]
                atom_bag, coords, Z = get_molcoords(molecule)
                heavy_atom_count = molecule.heavy_atom_count
                PE_bag = [1.0*mp.Element(i).X  for i in atom_bag]
                IR_bag = [1.0*mp.Element(i).average_ionic_radius for i in atom_bag]
                AR_bag = [1.0*mp.Element(i).atomic_radius for i in atom_bag]
                OX_bag = [1.0*mp.Element(i).common_oxidation_states[0] for i in atom_bag]
            else:
                atom_bag = df_smiles[(df_smiles['cids'] == molecule_name )]['atom_bag_list'].values[0]
                heavy_atom_count = df_smiles[(df_smiles['cids'] == molecule_name )]['heavy_atom_counts'].values[0]
                coords = df_smiles[(df_smiles['cids'] == molecule_name )]['coords_list'].values[0]
                coords = coords.replace('[','')
                #convert coords from string to numpy 2d-array
                coords = coords.replace(']','')
                coords = np.fromstring(coords, sep=',')#.reshape(-1, 4)
                coords = coords.reshape(-1,3)
                #
                Z = df_smiles[(df_smiles['cids'] == molecule_name )]['Z_list'].values[0]
                Z = Z.replace('[','')
                Z = Z.replace(']','')
                Z = np.fromstring(Z, sep=',')#.reshape(-1, 4)
                if type(atom_bag) == str:
                    atom_bag = str_to_list(atom_bag)
                PE_bag = [1.0*mp.Element(i).X  for i in atom_bag]
                IR_bag = [1.0*mp.Element(i).average_ionic_radius  for i in atom_bag]
                AR_bag = [1.0*mp.Element(i).atomic_radius  for i in atom_bag]
                OX_bag = [1.0*mp.Element(i).common_oxidation_states[0] for i in atom_bag]

        N = len(atom_bag)
        if N > 0:
            MolMatrix = np.zeros((NUM_flavors + 1,N,N),dtype=object)
            # Use pymatgen_molecue class
            pm_molecule = Molecule(atom_bag, coords)
            mol_dm = pm_molecule.distance_matrix
            bond_list = []
            for i in np.arange(N):
                atom_i = atom_bag[i]
                for j in np.arange(N):
                    if j >= i:
                        atom_j = atom_bag[j]
                        if i == j:
                            bondpair = atom_i
                            bond_list.append(bondpair)
                            MolMatrix[0,i,j] = bondpair
                            MolMatrix[1,i,j] = 0.5*Z[i]**2.4
                            MolMatrix[2,i,j] = PE_bag[i]
                            MolMatrix[3,i,j] = IR_bag[i]
                            MolMatrix[4,i,j] = AR_bag[i]
                            MolMatrix[5,i,j] = OX_bag[i]
                        else:
                            bondpair = atom_i + atom_j
                            bond_list.append(bondpair)
                            MolMatrix[1,i,j] = Z[i]*Z[j]/mol_dm[i,j]
                            MolMatrix[0,i,j] = bondpair
                            MolMatrix[2,i,j] = np.abs(PE_bag[i]*Z[i]-PE_bag[j]*Z[j])/mol_dm[i,j]
                            MolMatrix[3,i,j] = np.abs(IR_bag[i]-IR_bag[j])/mol_dm[i,j]
                            MolMatrix[4,i,j] = np.abs(AR_bag[i]-AR_bag[j])/mol_dm[i,j]
                            MolMatrix[5,i,j] = np.abs(OX_bag[i]*1.0-OX_bag[j]*1.0)/mol_dm[i,j]
                    MolMatrix[0,j,i] = MolMatrix[0,i,j]
                    MolMatrix[1,j,i] = MolMatrix[1,i,j]
                    MolMatrix[2,j,i] = MolMatrix[2,i,j]
                    MolMatrix[3,j,i] = MolMatrix[3,i,j]
                    MolMatrix[4,j,i] = MolMatrix[4,i,j]
                    MolMatrix[5,j,i] = MolMatrix[5,i,j]
            uniquepair = np.unique(bond_list)

        else:
            MolMatrix = np.nan
        for flav in np.arange(NUM_flavors-1):
            if heavy_atom_count == 0:
                MolMatrix[flav+1,:,:] = MolMatrix[flav+1,:,:]/0.5
            else:
                MolMatrix[flav+1,:,:] = MolMatrix[flav+1,:,:]/np.float(heavy_atom_count)
    return MolMatrix, atom_bag


def get_bob_cs(molecule_list, sigma, min_dist, p, on_the_fly,df_smiles, NUM_flavors):
    """
        calculates chemical space values for a list of molecules
        - fth component in the first index for features_flavors
        - for targeted testing of groups of molecules
    """
    Bob_vector_flavors, atoms_list = get_BoBvector_cids_multi(molecule_list, 'cid', min_dist, on_the_fly,df_smiles,NUM_flavors)
    features_flavors = get_Bobfeatures_multi(Bob_vector_flavors, sigma,p)
    return features_flavors
