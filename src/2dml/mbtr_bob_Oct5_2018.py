# py file for functions written is MBTR_catalysis
# For use with catalysis.ipynb file

# COPIED TO MBTR.PY JUN24 2018 6:20pm
# Updated June 30, 2018
# copied from catalysis_v19 jul8 after possible changes

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
from pymatgen import Lattice, Structure, Molecule
import re
import pymatgen as mp
import functools
import abx_vector as abx
# NOTES:
# - consider i=j case separatelu
# - account for double counting of entires of off-diagonal terms
from mendeleev import element




# def gen_abx_bag(myspecies):
#     """
#         create atom_bag - equivalent to get_molcoords()
#     """
#     myspecies = mp.Composition(myspecies)
#     abx_formula = abx.abx_vector_gen(myspecies) #pymatgen elements
#     abx_bag  = [str(x) for x in abx_formula] #strings
#     return abx_bag

def gen_abx_bag(myspecies):
    """
        create atom_bag - equivalent to get_molcoords()
    """
    myspecies = mp.Composition(myspecies)
    df2 = pd.DataFrame() #create dummy dataframe to initialize property_vector()
    abxvec = abx.property_vector(df2)
    abx_formula = abxvec.abx_vector_gen(myspecies) #pymatgen elements
    abx_bag  = [str(x) for x in abx_formula] #strings
    return abx_bag

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
            atom_bag, coords, Z = gen_abx_bag(molecule)
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
        #print len(atom_bag), type(atom_bag)
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
                        #print bondpair, type(bondpair)
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
        #print 'unique', uniquepair
        #print '\n MOL MATRIX: \n \n', MolMatrix
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
            bond_index = np.argwhere(MolMatrix[0,:,:] == bond)
            #num_atoms = MolMatrix.shape[1]
            #num_nn = (len(np.argwhere(MolMatrix[1,:,:] <= min_dist))-num_atoms)/2.0
            #print bond_index
            bond_counts = [] #for a given bondtype get the list of matrix_Values
            for b_dex in bond_index:
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



def get_master_set(molecule_list, min_dist,identifier,df_smiles):
    """
        Defined in mbtr.py
        Create a set of MasterMatrices from the data
        """
    master_list = get_masterlist(molecule_list, df_smiles) #this is called with fewer arguments than needed
    #get_masterlist(molecule_list, on_the_fly, df_smiles, NUM_flavors)
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
    vec_length = master_matrix_set[0].shape[0]*master_matrix_set[0].shape[1]
    BoB_vector = np.zeros((vec_length, set_size))
    for mth, mat_item in enumerate(master_matrix_set):
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
    master_matrix_set, atom_bag_list = get_master_set_multi(molecule_list, on_the_fly,df_smiles)
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
        df2b = pd.concat([df2b,df_prints], axis=1, ignore_index=False)
    return df2b

# xxx

filestem = '/Users/trevorrhone/Documents/Kaxiras/FRiend/catalysis_April_2018/'

def bob_update(df2, sigma, min_dist,on_the_fly,p,df_smiles,NUM_flavors,filestem = filestem):
    """
        Updates bob chemical space values for r1_bob, p1_bob, etc.
    """
    # Add sub_fingerprint info to dataframe for the species_list:
    df2b = df2.copy()
    # *****
    # Calculate results and Create pickled file if does not exists:
    # get_bob_df(df2b)
    recalculate = True
    filepath = filestem + 'bob_space.p'
    if not os.path.exists(filepath) or recalculate:
        print('recalculating bob_df')
        df_bob_list, rth_bob_space,atoms_list = get_bob_multi(df2b,sigma,min_dist,on_the_fly,p,df_smiles,NUM_flavors)
        #print('df_bob_list', df_bob_list)
        #update df after calculating bob chem space
        for fth in np.arange(NUM_flavors):
            #first bob_space index is for reactant/prodcut.. second index is for chem space flavor
            #print('df_bob_list[fth]', df_bob_list[fth])
            for rth, val in enumerate(df_bob_list[fth]):
                #print('rth', rth)
                #print(len(rth_bob_space[rth][fth]))
                df2b[val] = rth_bob_space[rth][fth]
            pickle.dump( rth_bob_space, open( "bob_space.p", "wb" ) )
    else:
        print('loading pickled file')
        rth_bob_space = pickle.load( open( "bob_space.p", "rb" ) )
        for fth in np.arange(NUM_flavors):
            #print('df_bob_list[fth]', df_bob_list[fth])
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


def get_Bobfeatures_multi(Bob_vector_flavors, sigma, p):
    """
        # Test implementatino by definning similarity metrix and plotting results
        # SHould use sigma as a tuning parameter
        # L_p norm delimited by p
        OUTPUT: set of f flavors of chemspace values of set of molecules of one of the reactants or producnts
    """
    #print('Bob_vector_flavors', Bob_vector_flavors)
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



def make_bag_multi(MolMatrix):
    """
        Make a bag - collection of bondtypes and corresponding property
        input: MolMatrix and minimum interatomic distances
        output: bondtypes, bag (i.e contents of each bag) , counts (i.e. lenght of each bag)
        This function fills the bag with the property of interatomic distance
    """
    if type(MolMatrix) == type(np.nan):
        #print 'get here'
        return [], [[],[],[],[],[]], 0
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
    return bondtypes, bag_set, counts



def get_bob_multi(df2, sigma, on_the_fly, p, df_smiles, NUM_flavors):
    """
        UPDATE DF WITH BOB FEATURE VECTORS
        NOTE:
        - First index is rth reactant/prodcut.. inner list index is flavor
        - in rth_bob_space
    """
    #reaction_list = ['Reactant 1 CID','Reactant 2 CID','Reactant 3 CID','Major Product CID']
    formula = 'formula'
    df_bob_list = [['cs_bob'],
                   ['cs_PE'],
                   ['cs_IR'],
                   ['cs_AR'],
                   ['cs_OX']]
    #rth_bob_space = []
    #atoms_reaction_list = []
    Bob_vector_flavors, atoms_list = get_BoBvectors_multi(df2, formula, on_the_fly, df_smiles, NUM_flavors)
    features_flavors = get_Bobfeatures_multi(Bob_vector_flavors, sigma, p)
    #atoms_reaction_list.append(atoms_list)
    #rth_bob_space.append(features_flavors)
    return df_bob_list, features_flavors, atoms_list



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



def get_BoBvectors_multi(df2,species, on_the_fly, df_smiles, NUM_flavors):
    """
        Get bob_vectors
        INPUT : 'Reactant X CID'. Converts CID to names using cid_to_name()
        - updated for ABX3 structures
    """
    molecule_list = df2[species]
    print('species',species)
    master_matrix_set, atom_bag_list = get_master_set_multi(molecule_list, on_the_fly, df_smiles, NUM_flavors)
    # CREATE BoB vectors from Master Matrices...
    set_size = len(master_matrix_set)
    vec_length = master_matrix_set[0].shape[1]*master_matrix_set[0].shape[2]
    Bob_vector = np.zeros((vec_length, set_size), dtype = np.float)
    Bob_vector_flavors = np.empty((NUM_flavors,vec_length, set_size), dtype = np.float)
    for fth in np.arange(NUM_flavors):
        for mth, mat_item in enumerate(master_matrix_set):
            vec_item = np.reshape(mat_item[fth,:,:], (1,-1), order = 'F')
            Bob_vector[:, mth] = vec_item
        Bob_vector_flavors[fth,:,:] = Bob_vector
    return Bob_vector_flavors, atom_bag_list



def get_masterlist(molecule_list, on_the_fly, df_smiles, NUM_flavors):
    """
        generates a master list of bondtypes (list of unique items) from the dataset
        - modify to work with ABX3 compounds
    """
    bondtypes_list = []
    #bag_list = []
    # Defune molecule & adjacency matrix
    for molecule_name in molecule_list:
        #molecule_name = molname
        #print 'molecule_name', molecule_name
        MolMatrix, atom_bag = mol_matrix_multi(molecule_name, on_the_fly, df_smiles, NUM_flavors)
        bondtypes, bag, bond_counts = make_bag_multi(MolMatrix)
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
    reduced_list = functools.reduce(operator.concat, combined_list)
    master_list = np.unique(reduced_list)
    #print 'master list', master_list
    return master_list






def get_master_set_multi(molecule_list, on_the_fly, df_smiles, NUM_flavors):
    """
        Defined in mbtr.py
        Create a set of MasterMatrices from the data
        updates: update to only accept CIDS numbers...
        - update to include ABX3 type
    """
    master_list = get_masterlist(molecule_list, on_the_fly, df_smiles, NUM_flavors)
    master_matrix_set = []
    master_matrix_flavors = []
    atom_bag_list = []
    for molname in molecule_list:
        #print('molname',np.int(molname))
        MolMatrix, atom_bag = mol_matrix_multi(molname,on_the_fly,df_smiles,NUM_flavors)
        bondtypes, bag_set, bond_counts = make_bag_multi(MolMatrix)
        #print(atom_bag)
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
    #print (unique_names)
    #             get_BoBvectors_debug(molecule_list, species, identifier, min_dist, on_the_fly):
    smilesvectors, atoms_list = get_BoBvector_cids_multi(unique_names,'cid',min_dist,on_the_fly,df_smiles,NUM_flavors)
    smilesbobs = get_Bobfeatures_multi(smilesvectors,sigma,p)
    return smilesbobs




def create_smilesbobs(smilesRECALCULATE, df_smiles,sigma, min_dist,on_the_fly,p, filestem = filestem):
    """
        creates smiles bobs from df_smiles
        """
    if smilesRECALCULATE == True:
        ## throttle_smilesbobs:
        dsize =  500
        N = len(df_smiles)
        print('Total # of SMILES: ', N)
        steps = N/dsize*1.0
        #print '# of steps', steps
        numsteps = np.floor(steps)

        step_sequence = np.arange(0,numsteps*dsize,dsize)
        step_sequence = np.asarray(step_sequence, dtype = int)
        #print step_sequence[:5]
        smilesbobs_list = []

        for ith, i in enumerate(step_sequence[:]):
            #print i, i+dsize
            sub_df_smiles = df_smiles[i:i+dsize]
            sub_smilesbobs = get_smilesbobs(sub_df_smiles, sigma, min_dist,on_the_fly,p,NUM_flavors)
            smilesbobs_list.append(sub_smilesbobs)
        #time.sleep(90)
        sub_df_smiles = df_smiles[step_sequence[-1] + dsize: N]
        sub_smilesbobs = get_smilesbobs(sub_df_smiles, sigma,min_dist,on_the_fly,p,NUM_flavors)
        smilesbobs_list.append(sub_smilesbobs)

        pickle.dump( smilesbobs_list, open( "smilesbobs_list.p", "wb" ) )
        bobfileloc = filestem + 'smilesbobs_saved.csv'

        #####
        for i, s in enumerate(smilesbobs_list):
            if i == 0:
                smilesconcat = s
            else:
                smilesconcat = np.concatenate((smilesconcat,s),axis=1)
        #print 'type s' , type(s), s.shape
        #print 'smilesconcat.shape', smilesconcat.shape

        df_smilesbobs = pd.DataFrame(smilesconcat)
        #np.savetxt(bobfileloc, smilesbobs, fmt='%.18e', delimiter = ',')
        #THESE ARE NOT 'UNPACKED SMILESBOBS... EACH ROW REPRESENTS ITH STEP_SEQUENCE RESULT
        df_smilesbobs.to_csv(bobfileloc, index=True, mode='w', sep=',')
        pickle.dump( smilesconcat, open( "smilesbobs.p", "wb" ) )
        #print smilesconcat.shape
        smilesbobs = smilesconcat #reset label for agreement wiht the rest of the code you've written
    return smilesbobs


def getMasterMatrix_multi(bag_set, bond_counts, master_list, bondtypes, NUM_flavors):
    """
        Fill Master Matrix with values vased on master_list
       """
    #print bag_set, '\n'
    bag = bag_set[0]
    #print 'bag', bag
    bag_PE = bag_set[1]
    bag_IR = bag_set[2]
    bag_AR = bag_set[3]
    bag_OX = bag_set[4]
    #print 'bag ir', bag_IR
    # Create master matrix:
    if type(bag) == type(np.nan):
        return np.nan
    #max_rowsize = np.max(bond_counts) # THIS VARIES depending on molecule..
    max_rowsize = 2000                  # Want to maximize this.
    # That is, consider for biggest molecule
    max_colsize = len(master_list)
    MasterMatrix = np.zeros((NUM_flavors, max_rowsize, max_colsize),dtype=np.float)
    for ith, bth in enumerate(master_list):
        # print bth, ith
        # bdex = in_list(bth,bondtypes)
        bdex = np.argwhere(bondtypes == bth)
        # print bdex
        if len(bdex) == 0:
            #print 'SHIOT', sizeentry, ith
            #print max_rowsize, MasterMatrix.shape
            MasterMatrix[0,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[1,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[2,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[3,:,ith] = np.zeros(max_rowsize, dtype=np.float)
            MasterMatrix[4,:,ith] = np.zeros(max_rowsize, dtype=np.float)
        else:
            bdex = bdex[0][0]#-1 #count list from 1st element but array from 0th..
            #print 'bdex', bdex
            bag_element = bag[bdex]
            bag_PE_element = bag_PE[bdex]
            bag_IR_element = bag_IR[bdex]
            bag_AR_element = bag_AR[bdex]
            bag_OX_element = bag_OX[bdex]
            #
            #print 'bag_element ', bag_element, '\n'
            sizeentry = len(bag_element)
            #print 'MasterMatrix.shape', MasterMatrix.shape, 'bedex', bdex
            #print 'length bag_element ', sizeentry
            MasterMatrix[0, :sizeentry, bdex] = bag_element
            MasterMatrix[1, :sizeentry, bdex] = bag_PE_element
            MasterMatrix[2, :sizeentry, bdex] = bag_IR_element
            MasterMatrix[3, :sizeentry, bdex] = bag_AR_element
            MasterMatrix[4, :sizeentry, bdex] = bag_OX_element
    return MasterMatrix



def make_tree_bob_space(df2,df_smiles_nna,constrain,use_weight):
    """
        location: mbtr.py
        Use df_unique_smiles as dictionary to look up values of df2['product tree smiles']
        - allows to use wither 'possible_products' or 'constrain_prodcuts'
        - have to specify target cs type
    """
    bob_bag = []
    target_cs = 'smilebobs_OX'
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
        #print('smiles_bag', smiles_bag)
        #loop through a list of possible products in a given row of df2 (ie a given reaction)
        if len(np.atleast_1d(smiles_bag)) == 1:
            #print(smiles_bag, type(smiles_bag))
            #
            # Account for single item not in list (i.e. 'N')
            if type(smiles_bag) != type(list()):
                smiles_bag = [smiles_bag]
                #print(smiles_bag, type(smiles_bag))
            sm_index = df_smiles_nna.index[df_smiles_nna.loc[:,'smiles'] == str(smiles_bag[0])]
            #bob = df_smiles_nna.loc[sm_index,'smilebobs_OX'].values[0]
            #print('str(smiles_bag[0])',str(smiles_bag[0]))
            #print('sm_index', sm_index)
            # Account for empty sm_index
            if len(sm_index) > 0:
                bob = df_smiles_nna.loc[sm_index, target_cs].values[0]
            else:
                print('******** got to np.nan')
                bob = np.nan
            #print('smiles bag len 1 -- bob', bob)
            bob_set.append(bob)
        else:
            #print('smiles_bag   ---',smiles_bag)
            for smile in smiles_bag:
                #print('smile ', smile)
                #sm_index = np.where(smile == df_smiles_nna['smiles'])
                sm_index = df_smiles_nna.index[df_smiles_nna.loc[:,'smiles'] == smile]
                print(sm_index)
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
                    #print(bob, type(bob))
                    bob_set.append(bob)
                else:
                    print('got second part np.nan')
                    print('smie',smile)
                    bob_set.append(np.nan)
        bob_bag.append(bob_set)
    return bob_bag




def mol_matrix_multi(structure,on_the_fly,df_smiles,NUM_flavors):
    """
        returms an adjacency matrix, mol_matrix, that represents a molecule
        identifier can be 'name' or 'cids'-->  design up accepts CIDS
        updated: to get [atom_bag, coords, Z ] from df_smiles.csv and
        not from pcp.get_compounds()
        PE - Pauli electronegativity
        IR - ionic radius
        AR - atomic radius
        - modified to work with ABX3 com
    """
    if pd.isnull(structure) == True:
        #print 'is pd null'
        return np.nan, []
    elif structure == 0.0:    #checkpint
        #print 'is pd null'
        return np.nan, []
    else:
        if on_the_fly == True:
            #molecule_name = np.int(molecule_name)
            #molecule = pcp.get_compounds(molecule_name,identifier)[0]
            #heavy_atom_count = molecule.heavy_atom_count
            #print(type(structure),structure)
            atom_bag = gen_abx_bag(structure)
            Z =  [1.0*mp.Element(i).number  for i in atom_bag]
            PE_bag = [1.0*mp.Element(i).X  for i in atom_bag]
            # Pauling electronegativity. Elements without an electronegativity
            # number are assigned a value of zero by default.
            IR_bag = [1.0*mp.Element(i).average_ionic_radius  for i in atom_bag]
            AR_bag = [1.0*mp.Element(i).atomic_radius  for i in atom_bag]
            #OX_bag = [1.0*mp.Element(i).common_oxidation_states[0]  for i in atom_bag]
            OX_bag = [1.0*element(i).dipole_polarizability for i in atom_bag]
        else:
            if (df_smiles['cids'] == molecule_name ).any() == False:
                molecule = pcp.get_compounds(np.int(molecule_name),identifier)[0]
                atom_bag, coords, Z = gen_abx_bag(molecule)
                heavy_atom_count = molecule.heavy_atom_count
                PE_bag = [1.0*mp.Element(i).X  for i in atom_bag]
                IR_bag = [1.0*mp.Element(i).average_ionic_radius for i in atom_bag]
                AR_bag = [1.0*mp.Element(i).atomic_radius for i in atom_bag]
                OX_bag = [1.0*element(i).dipole_polarizability for i in atom_bag]
            else:
                atom_bag = df_smiles[(df_smiles['cids'] == molecule_name )]['atom_bag_list'].values[0]
                heavy_atom_count = df_smiles[(df_smiles['cids'] == molecule_name )]['heavy_atom_counts'].values[0]
                coords = df_smiles[(df_smiles['cids'] == molecule_name )]['coords_list'].values[0]
                coords = coords.replace('[','')
                # convert coords from string to numpy 2d-array
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
                #OX_bag = [1.0*mp.Element(i).common_oxidation_states[0] for i in atom_bag]
                OX_bag = [1.0*element(i).dipole_polarizability for i in atom_bag]
    N = len(atom_bag)
    if N > 0:
        MolMatrix = np.zeros((NUM_flavors + 1,N,N),dtype=object)
        # Use pymatgen_molecue class
        # pm_molecule = Molecule(atom_bag, coords)
        # mol_dm = pm_molecule.distance_matrix
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
                        #MolMatrix[5,i,j] = OX_bag[i]
                        #MolMatrix[5,i,j] = OX_bag[i]*1.0/(2.0*IR_bag[i])
                        MolMatrix[5,i,j] = 0.0 # gives INF
                    else:
                        bondpair = atom_i + atom_j
                        bond_list.append(bondpair)
                        MolMatrix[1,i,j] = Z[i]*Z[j]
                        MolMatrix[0,i,j] = bondpair
                        MolMatrix[2,i,j] = np.abs(PE_bag[i]*Z[i]-PE_bag[j]*Z[j])
                        MolMatrix[3,i,j] = np.abs(IR_bag[i]-IR_bag[j])
                        MolMatrix[4,i,j] = np.abs(AR_bag[i]-AR_bag[j])
                        dip_pol = np.abs(OX_bag[i]*1.0-OX_bag[j]*1.0)
                        rad_sum = IR_bag[i] + IR_bag[j]
                        MolMatrix[5,i,j] = (1.0*dip_pol)/(rad_sum) # 1/diip_pol gives inf
                MolMatrix[0,j,i] = MolMatrix[0,i,j]
                MolMatrix[1,j,i] = MolMatrix[1,i,j]
                MolMatrix[2,j,i] = MolMatrix[2,i,j]
                MolMatrix[3,j,i] = MolMatrix[3,i,j]
                MolMatrix[4,j,i] = MolMatrix[4,i,j]
                MolMatrix[5,j,i] = MolMatrix[5,i,j]
        uniquepair = np.unique(bond_list)
    else:
        MolMatrix = np.nan
    #     for flav in np.arange(NUM_flavors-1):
    #         if heavy_atom_count == 0:
    #             MolMatrix[flav+1,:,:] = MolMatrix[flav+1,:,:]/0.5
    #         else:
    #             MolMatrix[flav+1,:,:] = MolMatrix[flav+1,:,:]/np.float(heavy_atom_count)
    return MolMatrix, atom_bag






def get_bob_cs(molecule_list, sigma, min_dist, p, on_the_fly,df_smiles, NUM_flavors):
    """
        calculates chemical space values for a list of molecules
        - fth component in the first index for features_flavors
        - for targeted testing of groups of molecules
        """
    #NUM_flavors = 5
    Bob_vector_flavors, atoms_list = get_BoBvector_cids_multi(molecule_list, 'cid', min_dist, on_the_fly,df_smiles,NUM_flavors)
    #print Bob_vector_flavors
    features_flavors = get_Bobfeatures_multi(Bob_vector_flavors, sigma,p)
    return features_flavors
