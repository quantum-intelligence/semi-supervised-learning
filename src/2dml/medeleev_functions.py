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

import shutil


# py file for functions written is MBTR_catalysis
# For use with catalysis.ipynb file

# COPIED TO MBTR.PY SEPT21 6:20pm

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

from fmmlcalc_b import *






#
# COPIED FROM atomic_property
#







def makepretty_doping(mystr,rescale=False):
    """
        Converts formula input into a form materials project APS can understand
        * 1.22.2017 incorporate pymatgen get_reduced_formula_and_factor() to eliminate parens
        * 2.6.2017 updated code to correct issue with finding fraction respresentatin from decimal
        * 2.6.2017: tries to capture fractino of dopants where chemical formual would give constituents too
          large to be considered by materials project. convert '2-x' and 'x' to numbers
    """
    #print 'start func'
    #
    # Add to handle susceptibility data:
    #
    #mystr = dehydrate(mystr)
    #
    mystr = str(mystr)
    mystr = re.sub(r'\xa0','',mystr)
    #mystr = dehydrate(mystr)
    #print mystr
    #
    mystr = re.sub('OD','OH',mystr)
    mystr = re.sub('1\+y','',mystr)
    mystr = re.sub('\+y','',mystr)
    mystr = re.sub('\+d','',mystr)
    mystr = re.sub('Ky','K',mystr)
    mystr = re.sub('Rby','Rb',mystr)
    mystr = re.sub('-alpha','',mystr)
    mystr = re.sub('alpha-','',mystr)
    mystr = re.sub('\]n','',mystr)
    mystr = re.sub('\[','',mystr)
    mystr = re.sub('FeII','Fe',mystr)
    mystr = re.sub('III','',mystr)
    mystr = re.sub('2-x','1.9',mystr)
    mystr = re.sub('x','0.1',mystr)
    # Use pymatgen to eliminate brackets:
    # print 'before deaut', mystr
    mystr = deuterium(mystr)
    comp = mg.Composition(mystr)
    # print 'mg info', comp
    mystr_re = comp.formula
    #mystr_re = comp.get_reduced_formula_and_factor()
    # print 'my str', mystr
    # print 'reduced formula', mystr_re
    #comp_re = mg.Composition(mystr_re[0])
    #mystr_re = comp_re.get_reduced_formula_and_factor()
    #mystr = mystr_re[0]
    mystr = mystr_re.replace(" ","")
    pattern1 = r'[^\w.]'
    mystr = re.sub(pattern1, '', mystr)
    #print '2', mystr
    pattern2 = '[A-Z][a-z]?\d*.*d*'
    mystr = re.match(pattern2,mystr).group(0)
    mystr = re.sub('ND','NH',mystr) #convert deuterium to H sympbol
    mystr = re.sub('-','',mystr)
    liststr = re.findall('[A-Z][a-z]?|\d*\.?\d*', mystr)
    counter = -1
    #print 'my string', mystr
    #
    #UPDATE wtih function from Fraction class
    if rescale == True:
        scaling = make_rational(mystr)
        #print 'my scaling', scaling
        #
        #if there is some decimals that doesnt result in good
        #rescaling, ie scaling >>10, then jsut round the numbers.
        if scaling < 1000:
            for elem in liststr:
                counter = counter + 1
                u = unicode(elem.replace('.',''))
                isnum = u.isnumeric()
                if isnum == True:
                    newelem = float(elem)
                    newelem = newelem*scaling
                    newelem = str(int(newelem))
                    #newelem = str(int(round(newelem)))
                    if newelem == '1':
                        newelem = ''
                    liststr[counter] = newelem
        else:
            for elem in liststr:
                counter = counter + 1
                u = unicode(elem.replace('.',''))
                isnum = u.isnumeric()
                if isnum == True:
                    newelem = float(elem)
                    #newelem = newelem*scaling
                    #newelem = str(int(newelem))
                    newelem = str(int(round(newelem)))
                    if newelem == '1':
                        newelem = ''
                    liststr[counter] = newelem
        newstr = ''.join(liststr)
    else:
        newstr = mystr
    return newstr




def get_mendeleev(fname):
    """ works with addMGdata and call to m.get_data(formula)
        to extract a list of elements with their atomic masses.
        - uses call to element() in mendeleev package
        - should not not use presence of compound in materials project database
          as a means to assign values from mendeleev. Updated this on 1.17.2017
    """
    mol = 6.022140857e23
    atomicrad_list = []
    atomicvol_list = []
    covalentrad_list = []
    dipole_list = []
    eaffinity_list = []
    numelectron_list = []
    ionenergies_list = []
    oxi_list = []
    vdwradius_list = []
    en_list = []
    nvalence_list = []
    elem_list = []
    lastsubshell_list = []
    #
    boiling_point = []
    density = []
    evaporation_heat = []
    fusion_heat = []
    gas_basicity = []
    heat_of_formation = []
    melting_point = []
    thermal_conductivity = []
    #
    try:
        fdata = mg.Composition(fname)
    except ValueError:
        fdata = []
    if fdata == []:
        nanvector = np.ravel(np.zeros((1,14)))
        nanvector[:] = np.nan
        return nanvector
    #mpielements = fdata[0]['elements']
    mpielements = fdata.elements
    for elem in mpielements:
        elem_list.append(elem)
        elem = str(elem)
        # Get data from mendeleev
        x = element(elem)
        atomic_r = x.atomic_radius
        if atomic_r == None:
            pref = (4.0*np.pi)/3.0
            v = x.atomic_volume
            atomic_r = ((v/pref/mol)**(1./3.))*1e10
        atomic_v = x.atomic_volume
        cov_r = x.covalent_radius
        dipole = x.dipole_polarizability
        e_affinity = x.electron_affinity
        n_electrons = x.electrons
        ion_energies = x.ionenergies
        oxistates = x.oxistates
        vdw_radius = x.vdw_radius
        en_allen = x.en_allen
        nvalence = x.nvalence()
        lastsubshell = x.ec.last_subshell()
        boiling = x.boiling_point
        dense = x.density
        evaporation = x.evaporation_heat
        fusion = x.fusion_heat
        gas_basic = x.gas_basicity
        heat_of_form = x.heat_of_formation
        melting = x.melting_point
        thermal_conduct = x.thermal_conductivity
        atomicrad_list.append(atomic_r)
        atomicvol_list.append(atomic_v)
        covalentrad_list.append(cov_r)
        dipole_list.append(dipole)
        eaffinity_list.append(e_affinity)
        numelectron_list.append(n_electrons)
        ionenergies_list.append(ion_energies)
        oxi_list.append(oxistates)
        vdwradius_list.append(vdw_radius)
        en_list.append(vdw_radius)
        nvalence_list.append(nvalence)
        lastsubshell_list.append(lastsubshell)
        #
        boiling_point.append(boiling)
        density.append(dense)
        evaporation_heat.append(evaporation)
        fusion_heat.append(fusion)
        gas_basicity.append(gas_basic)
        heat_of_formation.append(heat_of_form)
        melting_point.append(melting)
        thermal_conductivity.append(thermal_conduct)
    # get fractional composition of elements in compound:
    # Needs to be in the appropriate order...
    y = mg.Composition(fname)
    frac_list = []
    for ith in elem_list:
        frac = y.get_atomic_fraction(ith)
        frac_list.append(frac)
    return (atomicrad_list, atomicvol_list, covalentrad_list, dipole_list, eaffinity_list,
            numelectron_list, ionenergies_list, oxi_list, vdwradius_list, en_list,
            nvalence_list, elem_list, frac_list,lastsubshell_list, boiling_point, density,
            evaporation_heat, fusion_heat, gas_basicity, heat_of_formation, melting_point,
            thermal_conductivity)





def get_mendeleev_data(df):
    """
        Gather data for formulae in a dataframe by using mendeleev package
        UPDATED 10.8.2017 : TO INCLUDE lists (eg atomicrad) where all elements have some length.
        in last version assignment to array would make an array of arrays, not array of lists.
    """
    N = len(df)
    atomicrad_list = np.empty(N,dtype=object)
    atomicvol_list = np.empty(N,dtype=object)
    covalentrad_list = np.empty(N,dtype=object)
    dipole_list = np.empty(N,dtype=object)
    eaffinity_list = np.empty(N,dtype=object)
    numelectron_list = np.empty(N,dtype=object)
    ionenergies_list = np.empty(N,dtype=object)
    oxi_list = np.empty(N,dtype=object)
    vdwradius_list = np.empty(N,dtype=object)
    en_list = np.empty(N,dtype=object)
    nvalence_list = np.empty(N,dtype=object)
    elem_list = np.empty(N,dtype=object)
    frac_list = np.empty(N,dtype=object)
    subshell_list = np.empty(N,dtype=object)
    boiling_point= np.empty(N,dtype=object)
    density= np.empty(N,dtype=object)
    evaporation_heat= np.empty(N,dtype=object)
    fusion_heat= np.empty(N,dtype=object)
    gas_basicity = np.empty(N,dtype=object)
    heat_of_formation = np.empty(N,dtype=object)
    melting_point = np.empty(N,dtype=object)
    thermal_conductivity = np.empty(N,dtype=object)
    #for ith, fname in df['formula'].iteritems():
    if 'level_0' in df:
        df.drop(columns='level_0')
    ## df_reset = df.reset_index(drop=True) -- modifield on 2.17.20220682
    ## ERROR: level_0 already exists error
    df_reset = df.reset_index(drop=True)
    for ith, fname in df_reset['formula'].iteritems():
        fname = makepretty_doping(fname) #need to eliminate white spaces and fractional constituents
        #atomicrad, atomicvol, covalentrad, dipole,eaffinity, numelectron, ionenergies, oxi,
        #vdwradius, en, nvalence = get_mendeleev(fname)
        #print(fname)
        #print(ith)
        mendel_value = get_mendeleev(fname)
        #print(mendel_value[0])
        #print(ith, len(atomicrad_list))
        atomicrad_list[ith] = (mendel_value[0])
        atomicvol_list[ith] = (mendel_value[1])
        covalentrad_list[ith] = (mendel_value[2])
        dipole_list[ith] = (mendel_value[3])
        eaffinity_list[ith] = mendel_value[4]
        numelectron_list[ith] = (mendel_value[5])
        ionenergies_list[ith] = (mendel_value[6])
        oxi_list[ith] = (mendel_value[7])
        vdwradius_list[ith] = (mendel_value[8])
        en_list[ith] = (mendel_value[9])
        nvalence_list[ith] = (mendel_value[10])
        elem_list[ith] = (mendel_value[11])
        frac_list[ith] = (mendel_value[12])
        subshell_list[ith] = (mendel_value[13])
        boiling_point[ith] = (mendel_value[14])
        density[ith] = (mendel_value[15])
        evaporation_heat[ith] = (mendel_value[16])
        fusion_heat[ith] = (mendel_value[17])
        gas_basicity[ith] = (mendel_value[18])
        heat_of_formation[ith] = (mendel_value[19])
        melting_point[ith] = (mendel_value[20])
        thermal_conductivity[ith] = (mendel_value[21])
    return (atomicrad_list, atomicvol_list, covalentrad_list, dipole_list, eaffinity_list,
            numelectron_list, ionenergies_list, oxi_list, vdwradius_list,
            en_list, nvalence_list, elem_list, frac_list,subshell_list,
            boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
            heat_of_formation, melting_point, thermal_conductivity)



def gen_mendel_data(df1, recalculate, filepath, data_name):
    """ wrapper for get_mendeleev_data code call and pickle load """

    # Calculate results and Create pickled file if does not exists:
    #recalculate = False
    picklefile = data_name + '.p'
    #filepath = '/Users/trevorrhone/Documents/Kaxiras/2DML/Shaan/mendeleevdata2.p'
    if not os.path.exists(filepath) or recalculate:
        #print('call get_mendeleev_data() and pickle rseults)
        (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
        ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,
        boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
        heat_of_formation, melting_point, thermal_conductivity) = get_mendeleev_data(df1)
        mendeleevdata2 = (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
                     ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,
                     boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
                     heat_of_formation, melting_point, thermal_conductivity)
        pickle.dump( mendeleevdata2, open( picklefile, "wb" ) )
    else:
        #print('loading pickled file')
        mendeleevdata2 = pickle.load( open( picklefile, "rb" ) )
        (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
        ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,
        boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
        heat_of_formation, melting_point, thermal_conductivity) = mendeleevdata2
    return (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron, ionenergies, oxi, vdwradius,
en, nvalence, elem_list, weights, lastshell, boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
heat_of_formation, melting_point, thermal_conductivity)


def build_df_mendel(df1, atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron, ionenergies, oxi, vdwradius,
    en, nvalence, elem_list, weights, lastshell, boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
    heat_of_formation, melting_point, thermal_conductivity):
    """ build up df using mendeleev data"""
    # mendeleev data:
    #
    (num_p, num_d, num_f, cmpd_mean_p, cmpd_mean_d, cmpd_mean_f,
     cmpd_dif_p, cmpd_dif_d, cmpd_dif_f) = shells(lastshell)
    df1['num_p'] = num_p
    df1['num_d'] = num_d
    df1['num_f'] = num_f
    df1['atomic_rad'] = atomicrad
    df1['atomic_vol'] = atomicvol
    df1['covalent_rad'] = covalentrad
    df1['dipole'] = dipole
    df1['eaffinity'] = eaffinity
    df1['num_electrons'] = numelectron

    df1['atomic_rad_avg'] = weighted_avg(atomicrad,weights)
    df1['atomic_rad_dif'] = max_difference(atomicrad)
    df1['atomic_vol_avg'] = weighted_avg(atomicvol,weights)
    df1['atomic_vol_dif'] = max_difference(atomicvol)
    df1['covalent_rad_avg'] = weighted_avg(covalentrad, weights)
    df1['covalent_rad_dif'] = max_difference(covalentrad)
    df1['dipole_avg'] = weighted_avg(dipole, weights)
    df1['dipole_dif'] = max_difference(dipole)
    df1['e_affinity_avg'] = weighted_avg(eaffinity, weights)
    df1['e_affinity_dif'] = max_difference(eaffinity)
    df1['num_electron_avg'] = weighted_avg(numelectron,weights)
    df1['num_electron_dif'] = max_difference(numelectron)

    df1['vdw_radius_avg'] = weighted_avg(vdwradius,weights)
    df1['vdw_radius_dif'] = max_difference(vdwradius)
    df1['e_negativity_avg'] = weighted_avg(en,weights)
    df1['e_negativity_dif'] = max_difference(en)
    df1['n_valence_avg'] = weighted_avg(nvalence,weights)
    df1['n_valence_dif'] = max_difference(nvalence)
    df1['lastsubshell_avg'] = lastshell

    df1['cmpd_mean_p'] = cmpd_mean_p
    df1['cmpd_mean_d'] = cmpd_mean_d
    df1['cmpd_mean_f'] = cmpd_mean_f
    df1['cmpd_dif_p'] = cmpd_dif_p
    df1['cmpd_dif_d'] =  cmpd_dif_d
    df1['cmpd_dif_f'] =  cmpd_dif_f

    # include atomization energy of first and second atom
    # atomE_A = [x[0] for x in heat_of_formation]
    # atomE_B = [x[1] for x in heat_of_formation]
    atomE_AB = [x[1]+x[0] for x in heat_of_formation]
    df1['atomE_AB'] = atomE_AB #atomE_A + atomE_B
    Total_e = get_eTot(numelectron)
    frac_f = num_f/Total_e*1.0
    df1['frac_f '] = frac_f

    ion_list, dif_ion, sum_ion, mean_ion = get_ionization(ionenergies)
    df1['dif_ion'] = dif_ion
    df1['sum_ion'] = sum_ion
    df1['mean_ion'] = mean_ion
    # Add Born-Haber term
    # Born = dif_ion - max_difference(eaffinity) #this doesn't work for some reason
    Born = df1['dif_ion'].values - df1['e_affinity_dif'].values + df1['atomE_AB'] .values
    df1['Born'] = Born
    return df1



#def X2_gen(df_nna):
#     """
#         ssigns subset of features from df_nna to X2
#     """
#     X2 = df_nna[['energy', 'cohesive’,'num_p', 'num_d', 'num_f', 'atomic_rad', 'atomic_vol', 'covalent_rad','dipole','eaffinity','num_electrons','atomic_rad_avg','atomic_rad_dif', 'atomic_vol_avg', 'atomic_vol_dif','covalent_rad_avg', 'covalent_rad_dif', 'dipole_avg', 'dipole_dif','num_electron_avg', 'num_electron_dif', 'vdw_radius_avg', 'vdw_radius_dif','e_negativity_avg', 'e_negativity_dif', 'n_valence_avg','n_valence_dif', 'lastsubshell_avg', 'cmpd_mean_p', 'cmpd_mean_d','cmpd_mean_f', 'cmpd_dif_p', 'cmpd_dif_d', 'cmpd_dif_f', 'atomE_AB','frac_f ', 'dif_ion', 'sum_ion', 'mean_ion']]

#     target = X2[[u'cohesive']]

#     # Remove all product info :
#     # remove magmom since not so necessary and don't have this info for exo_data
#     drop_features = [['energy','cohesive', u'atomic_rad',u'atomic_vol', u'covalent_rad',u'dipole', u'eaffinity',u'num_electrons',u'lastsubshell_avg']]
#     for dropf in drop_features:
#         X2 = X2.drop(dropf, axis=1)

#     formulas = df_nna['formula']
#     return X2, target, formulas



def scaledata_xy(X2,y2_target):
    """
       No split & scale X and y_target data
    """
    y2 = y2_target
    #X2 = X2.drop('p_tanimoto',1)
    data2 = X2.copy()
    target2 = y2.copy()
    # convert pandas to numpy array:
    nptarget2 = target2.as_matrix(); nptarget2 = np.ravel(nptarget2);
    nptarget2 = np.ndarray.flatten(nptarget2)
    npdata2 = data2.as_matrix()
    #DO NOT SCALE target values
    #snptarget2 = preprocessing.scale(nptarget2)
    snptarget2 = nptarget2
    snpdata2 = preprocessing.scale(npdata2)
    return snpdata2, snptarget2



def scaledata(X2,y2_target,split,rstate):
    """
       Split then scale X and y_target data
    """
    y2 = y2_target
    data2 = X2.copy()
    target2 = y2.copy()
    # convert pandas to numpy array:
    # print type(target2)
    nptarget2 = target2.as_matrix(); nptarget2 = np.ravel(nptarget2);
    nptarget2 = np.ndarray.flatten(nptarget2)
    npdata2 = data2.as_matrix()
    # DO NOT SCALE TARGET DATA
    # snptarget2 = preprocessing.scale(nptarget2)
    snptarget2 = nptarget2
    snpdata2 = preprocessing.scale(npdata2)
    #X2scale = pd.DataFrame(snpdata2)
    X2_train, X2_test, y2_train, y2_test = cross_validation.train_test_split(snpdata2, snptarget2,
                                                        test_size=split, random_state=rstate)
    return X2_train, X2_test, y2_train, y2_test
