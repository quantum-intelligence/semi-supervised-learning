import sys
# code_path = '/Users/trevorrhone/Documents/highTc/dev/tc_codes'
# sys.path.append(code_path)

import numpy as np
import pandas as pd
import pymatgen as mp
import pickle
import os
from pymatgen.io.cif import CifParser

import wget
from os import path
import time
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import manifold
from sklearn.manifold import TSNE
import timeit
import pymatgen.analysis.local_env as lenv
import mysql.connector
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
import plotly.express as px

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

import pandas as pd
from random import sample
from tensorflow.keras.utils import to_categorical
from sklearn.decomposition import PCA

import importlib #reload

import tensorflow as tf
import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
tf.disable_v2_behavior()

from ase import Atoms
from sklearn import preprocessing

import NN_codes
importlib.reload(NN_codes)
from NN_codes import *
from mendeleev import element
# from ase.utils import reader, writer
from pymatgen.io.ase import AseAtomsAdaptor


def parse_spdf(spdf):
    s = []
    p = []
    d = []
    f = []
    for x in spdf:
        s.append(x[0])
        p.append(x[1])
        d.append(x[2])
        f.append(x[3])
    return s, p, d, f


def get_atomic_property(row_elems, loadresult, path):
    """
     get atomic property
     - does not use saved data from dictionary
    """
    # mendel_elems = [] #slower
    mendel_elems = np.zeros(len(row_elems), dtype=object)
    # print("get_atomic_property: row_elems",row_elems)
    for ith, x in enumerate(row_elems):
        #print('x in row elems', x)
        try:
            row_elem_name = x.name
            #print("row_elem_name",row_elem_name, "in get_atomic_property")
            atom_string, mendel_elem = atom_lookup(row_elem_name, loadresult, path)
        except:
            mendel_elem = np.nan
        # mendel_elems.append(mendel_elem) #slower
        mendel_elems[ith] = mendel_elem
    #row_elems_list = [element(x.name) for x in row_elems]
    #row_elems = []
    #row_elem = element(x.name)
    return mendel_elems

def atom_lookup(row_elem, loadresult,path):
    """
    save mendel data to dictionary and
    pull it from dictionry if it exists
    """
    #picklename = "mendel_dict.p"
    #picklepath = os.path.join(path,picklename)
    #saveresult = False
    #loadresult = False #DASK doesn't one to save things in multithreading mode
    #if loadresult:
    #print(' get pickled data')
    #atomdict = pickle.load( open( picklepath, "rb" ) )
    #else:
    atomdict = {}
    key = row_elem
    #print('key', key, type(key))
    if key not in atomdict:
        #print('not in dict, add ---', key)
        #saveresult = True ## DO NOT READ/READ with DASK
        value = element(key)
        #print('value', value, type(value))
        atomdict[key] = value
    else:
        #print('pull from dict')
        value = atomdict[key]
    #if saveresult:
    #    #print(' download files')
    #    pickle.dump( atomdict, open( picklepath, "wb" ) )
    return key, value

#@delayed(nout=2)   #delay this
def gen_properties(dftest_remc, ith, desc, loadresult, path):
    """ create list of propoerties for a cmpd """
    row = dftest_remc['formula'][ith]
    print(" gen_properties, row", row)
    row_cmpd = mp.Composition(row)
    row_elems_ = row_cmpd.elements
    # print('row_elems in gen_properties', row_elems_)
    row_elems = get_atomic_property(row_elems_, loadresult, path)
    # atom_lookup(row_elems, True, stem)
    #p_list = []
    p_list = np.zeros(len(row_elems))
    for ith, elem in enumerate(row_elems):
        #print('getattr', type(elem), elem)
        try:
            p = getattr(elem, desc)
            if p is None:
                p = np.nan
        except:
            #try:
            #p = getattr(elem[0], desc)
            #except:
            p = np.nan
        #p_list.append(p)
        p_list[ith] = p
    p_list = np.asarray(p_list)
    return p_list, row_cmpd


#@delayed(nout=2) #delay this
def get_tot_and_fraction(row_cmpd):
    """
        used pymatgen to parse atomic fraction
        and total number of electrons in a compound
    """
    row_elems_ = row_cmpd.elements
    #print("row_elems_", row_elems_)
    try:
        #elem_str = [elem.name for elem in row_elems_]
        #atomic_fractions = [row_cmpd.get_atomic_fraction(x) for x in elem_str]
        #total_electrons = row_cmpd.total_electrons
        N = len(row_elems_)
        elem_str = np.zeros(N, dtype=object)
        for ith, elem in enumerate(row_elems_):
            elem_str[ith] = elem.name
        n = len(elem_str)
        atomic_fractions = np.zeros(n, dtype=object)
        for ith, x in enumerate(elem_str):
            atomic_fractions[ith] = row_cmpd.get_atomic_fraction(x)
        total_electrons = row_cmpd.total_electrons
    except:
        atomic_fractions = np.nan
        total_electrons = np.nan
    #results = list(zip(atomic_fractions, total_electrons))
    return atomic_fractions, total_electrons

def gen_Stot(row_cmpd, unpaired_list):
    """
    # convert unpaired_list to Stot
    # gen_Stot()
    # Need to check if calculatino of Stot is reasonable!!
    # take into account the valence of the Cr3+ etc?
    """
    #print(row_cmpd, unpaired_list)
    elems = mp.Composition(row_cmpd)
    #print(elems, type(elems),list(elems))
    A_or_B_site = list(elems)[1]
    A2_is_metal = A_or_B_site.is_metal
    if A2_is_metal:
        #print("a2 and a1 metals exist")
        s = np.mean(unpaired_list[:2])
        S_tot = np.sqrt(s*(s+1))
    else:
        #print("only one A type o metal")
        s = unpaired_list[0]
        S_tot = np.sqrt(s*(s+1))
    return S_tot


def gen_P_electronic_cmpd_list(dftest_remc, loadresult, stem):
    """
    create P given p list
        - outlerloop descriptors
    - inner loop df row
    - generate P_list only ( dont do mean and std calc yet )
    """
    N = len(dftest_remc['formula'])
    S_list= np.zeros(N, dtype=object)
    P_list= np.zeros(N, dtype=object)
    D_list= np.zeros(N, dtype=object)
    F_list= np.zeros(N, dtype=object)
    Val_list = np.zeros(N, dtype=object)
    Tot_e_list = np.zeros(N, dtype=object)
    Unpaired_list = np.zeros(N, dtype=object)
    Stot_list = np.zeros(N, dtype=object)
    for ith in np.arange(N):
        #print("dftest_remc['formula'][ith]", dftest_remc['formula'][ith])
        eprop = gen_electronic_properties(dftest_remc, ith, loadresult, stem)
        #print("eprop", eprop)
        spdf_list, val_list, tot_e_list, unpaired_list, row_cmpd = eprop
        #print(spdf_list, val_list, tot_e_list, unpaired_list, row_cmpd )
        s_list, p_list, d_list, f_list = parse_spdf(spdf_list)
        #print("pares", s_list, p_list, d_list, f_list )
        Stot = gen_Stot(row_cmpd, unpaired_list)
        # frac, tot_e = get_tot_and_fraction_nodask(row_cmpd)
        # print("row_cmpd", row_cmpd)
        frac, tot_e = get_tot_and_fraction(row_cmpd)
        #print("get tot", frac, tot_e )
        try:
            s_list = np.asarray(frac)*np.asarray(s_list) #use weighted sum
            p_list = np.asarray(frac)*np.asarray(p_list) #use weighted sum
            d_list = np.asarray(frac)*np.asarray(d_list) #use weighted sum
            f_list = np.asarray(frac)*np.asarray(f_list) #use weighted sum
            val_list = np.asarray(frac)*np.asarray(val_list) #use weighted sum
            tot_e_list = np.asarray(frac)*np.asarray(tot_e_list) #use weighted sum
            unpaired_list = np.asarray(frac)*np.asarray(unpaired_list) #use weighted sum
        except:
            #print('got excet. frac?',frac,s_list,val_list,unpaired_list)
            s_list = np.nan; p_list = np.nan; d_list = np.nan; f_list = np.nan
            val_list = np.nan
            tot_e_list = np.nan
            unpaired_list = np.nan
        #P_list.append(weighted_p_list)
        S_list[ith] = s_list
        P_list[ith] = p_list
        D_list[ith] = d_list
        F_list[ith] = f_list
        Val_list[ith] = val_list
        Tot_e_list[ith] = tot_e_list
        Unpaired_list[ith] = unpaired_list
        Stot_list[ith] = Stot
    list_list = S_list, P_list, D_list, F_list, Val_list, Tot_e_list, Unpaired_list, Stot_list
    return list_list


def get_valence(elem):
    #print("get_valence", elem)
    try:
        valence = elem.nvalence()
    except:
        return np.nan
    return valence

def get_total_electrons(elem):
    try:
        tot_e = elem.electrons
    except:
        return np.nan
    return tot_e

def get_unpaired(elem):
    try:
        unpaired_e = elem.ec.unpaired_electrons()
    except:
        return np.nan
    return unpaired_e

def get_spdf(elem):
    """ get spdf shells """
    try:
        elem_conf = elem.ec.conf
    except:
        return np.nan, np.nan, np.nan, np.nan
    # elem_conf.items()
    # elem_conf.keys()
    s_shells = 0
    p_shells = 0
    d_shells = 0
    f_shells = 0
    # print('elem_conf', elem_conf)
    for orbital in elem_conf:
        n = orbital[0]
        l = orbital[1]
        #print(n,l)
        if l == 's':
            s_shells += 1
        elif l == 'p':
            p_shells += 1
        elif l =='d':
            d_shells += 1
        else:
            f_shells += 1
    return s_shells, p_shells, d_shells, f_shells



def gen_electronic_properties(dftest_remc, ith, loadresult, path):
    """ create list of propoerties for a cmpd """
    row = dftest_remc['formula'][ith]
    row_cmpd = mp.Composition(row)
    #print("row_cmpd", row_cmpd)
    row_elems_ = row_cmpd.elements
    #print("row_elems_",row_elems_)
    row_elems = get_atomic_property(row_elems_, loadresult, path)
    #print("row_elems",row_elems)
    N = len(row_elems)
    spdf_list = np.zeros(N, dtype=object)
    val_list = np.zeros(N, dtype=object)
    tot_e_list = np.zeros(N, dtype=object)
    unpaired_list = np.zeros(N, dtype=object)
    for ith, elem in enumerate(row_elems):
        #print(ith, elem)
        spdf = get_spdf(elem)
        val = get_valence(elem)
        tot_e = get_total_electrons(elem)
        unpaired = get_unpaired(elem)
        spdf_list[ith] = spdf
        val_list[ith] = val
        tot_e_list[ith] = tot_e
        unpaired_list[ith] = unpaired
    #print(spdf_list, val_list, tot_e_list, unpaired_list)
    return spdf_list, val_list, tot_e_list, unpaired_list, row_cmpd
