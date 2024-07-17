#
# Modified for python3.6 version
#

from math import floor, ceil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ase.db
import seaborn as sns

from matplotlib import cm
import pickle
import os.path

# from sklearn import cross_validation
from sklearn.model_selection import *
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import pymatgen as mg
from mendeleev import element
from fractions import Fraction

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

#fprintdata = np.load('fprintdata.npy')
#bandg = np.load('bandg.npy')

from fmcalc import *
from fmmlcalc_b import *



KRRkernel = 'rbf'

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
    df_reset = df.reset_index()
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



def update_mendel(exodata_sub,recalculate=True):
    """
        update a dataframe with features from mendeleev database
    """
    # Calculate results and Create pickled file if does not exists:
    # recalculate = True
    filepath = '/Users/trevorrhone/Documents/Kaxiras/2DML/Shaan/mendeleev_exo.p'
    if not os.path.exists(filepath) or recalculate:
        print('call getallinfo() and pickle rseults')
        (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
        ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,boiling_point,
        density, evaporation_heat, fusion_heat, gas_basicity, heat_of_formation, melting_point,
        thermal_conductivity) = get_mendeleev_data(exodata_sub)
        mendeleev_exo = (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
                     ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,
                     boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
                     heat_of_formation, melting_point, thermal_conductivity)
        pickle.dump( mendeleev_exo, open( "mendeleev_exo.p", "wb" ) )
    else:
        print('loading pickled file')
        mendeleev_exo = pickle.load( open( "mendeleev_exo.p", "rb" ) )
        (atomicrad, atomicvol, covalentrad, dipole, eaffinity, numelectron,
        ionenergies, oxi, vdwradius, en, nvalence, elem_list, weights, lastshell,
        boiling_point, density, evaporation_heat, fusion_heat, gas_basicity,
        heat_of_formation, melting_point, thermal_conductivity) = mendeleev_exo

    (num_p, num_d, num_f, cmpd_mean_p, cmpd_mean_d, cmpd_mean_f,
     cmpd_dif_p, cmpd_dif_d, cmpd_dif_f) = shells(lastshell)
    exodata_sub['num_p'] = num_p
    exodata_sub['num_d'] = num_d
    exodata_sub['num_f'] = num_f
    exodata_sub['atomic_rad'] = atomicrad
    exodata_sub['atomic_vol'] = atomicvol
    exodata_sub['covalent_rad'] = covalentrad
    exodata_sub['dipole'] = dipole
    exodata_sub['eaffinity'] = eaffinity
    exodata_sub['num_electrons'] = numelectron

    # mendeleev data:
    exodata_sub['atomic_rad_avg'] = weighted_avg(atomicrad,weights)
    exodata_sub['atomic_rad_dif'] = max_difference(atomicrad)
    exodata_sub['atomic_vol_avg'] = weighted_avg(atomicvol,weights)
    exodata_sub['atomic_vol_dif'] = max_difference(atomicvol)
    exodata_sub['covalent_rad_avg'] = weighted_avg(covalentrad, weights)
    exodata_sub['covalent_rad_dif'] = max_difference(covalentrad)
    exodata_sub['dipole_avg'] = weighted_avg(dipole, weights)
    exodata_sub['dipole_dif'] = max_difference(dipole)
    exodata_sub['e_affinity_avg'] = weighted_avg(eaffinity, weights)
    exodata_sub['e_affinity_dif'] = max_difference(eaffinity)
    exodata_sub['num_electron_avg'] = weighted_avg(numelectron,weights)
    exodata_sub['num_electron_dif'] = max_difference(numelectron)

    exodata_sub['vdw_radius_avg'] = weighted_avg(vdwradius,weights)
    exodata_sub['vdw_radius_dif'] = max_difference(vdwradius)
    exodata_sub['e_negativity_avg'] = weighted_avg(en,weights)
    exodata_sub['e_negativity_dif'] = max_difference(en)
    exodata_sub['n_valence_avg'] = weighted_avg(nvalence,weights)
    exodata_sub['n_valence_dif'] = max_difference(nvalence)
    exodata_sub['lastsubshell_avg'] = lastshell

    exodata_sub['cmpd_mean_p'] = cmpd_mean_p
    exodata_sub['cmpd_mean_d'] = cmpd_mean_d
    exodata_sub['cmpd_mean_f'] = cmpd_mean_f
    exodata_sub['cmpd_dif_p'] = cmpd_dif_p
    exodata_sub['cmpd_dif_d'] =  cmpd_dif_d
    exodata_sub['cmpd_dif_f'] =  cmpd_dif_f

    atomE_AB = [x[1]+x[0] for x in heat_of_formation]
    exodata_sub['atomE_AB'] = atomE_AB #atomE_A + atomE_B
    Total_e = get_eTot(numelectron)
    frac_f = num_f/Total_e*1.0
    exodata_sub['frac_f '] = frac_f

    ion_list, dif_ion, sum_ion, mean_ion = get_ionization(ionenergies)
    exodata_sub['dif_ion'] = dif_ion
    exodata_sub['sum_ion'] = sum_ion
    exodata_sub['mean_ion'] = mean_ion
    #Add Born-Haber term
    #Born = dif_ion - max_difference(eaffinity) #this doesn't work for some reason
    Born = exodata_sub['dif_ion'].values - exodata_sub['e_affinity_dif'].values + exodata_sub['atomE_AB'].values
    exodata_sub['Born'] = Born

    return exodata_sub



def KRR_test_pred(X_krr,y2_target,X_exo, y_exo, best_gamma, best_lambda):
    """
        Input: Model training data - X_krr and y2_target
               Model testing data - X_exo and y_exo
        Output: test prection, train set prediction
        location:
    """
    snpdatakrr, snptargetkrr = scaledata_xy(X_krr,y2_target)
    X_exo = scaledata_x(X_exo)
    #X2l_train, X2l_test, y2l_train, y2l_test = scaledata(X2_lasso,y2_target)
    runs = 1 #cant do more than one run if youre randomly splitting test and train set each time!!
    te_preds = []
    tr_preds = []
    for nth in np.arange(runs):
        rs = 4
        #Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = cross_validation.train_test_split(snpdatakrr, snptargetkrr,
        #                                       test_size=0.20, random_state=rs)
        Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = train_test_split(snpdatakrr, snptargetkrr,
                                              test_size=0.20, random_state=rs)
        clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
        #
        # better way to pick hyperparameters for KRR ???
        # don't have to fit on just train data if already chose hyperparam?
        clfkrr.fit(Xkrr_train, ykrr_train)
        krrte_pred = clfkrr.predict(X_exo)
        krrtr_pred = clfkrr.predict(snpdatakrr)
        tr_error = clfkrr.score(snpdatakrr, snptargetkrr)
        te_error = clfkrr.score(X_exo, y_exo)
        te_preds.append(krrte_pred)
        tr_preds.append(krrtr_pred)
    avg_te_pred = np.mean(te_preds,axis=0)
    avg_tr_pred = np.mean(tr_preds,axis=0)
    return avg_te_pred, avg_tr_pred, ykrr_test, ykrr_train


def scaledata_x(X2):
    """
       No split & scale X data only
    """
    data2 = X2.copy()
    #npdata2 = data2.as_matrix()
    npdata2 = data2
    snpdata2 = preprocessing.scale(npdata2)
    return snpdata2



def KRR_pred_all(X_krr,y2_target, best_gamma, best_lambda):
    """
        function for KRR
        Output: test prection for entire dataset.
        However ,model is trained on training set
    """
    # Use krr model to generate predictinos for a set of data
    snpdatakrr, snptargetkrr = scaledata_xy(X_krr,y2_target)
    # X2l_train, X2l_test, y2l_train, y2l_test = scaledata(X2_lasso,y2_target)
    runs = 1 #cant do more than one run if youre randomly splitting test and train set each time!!
    te_preds = []
    tr_preds = []
    for nth in np.arange(runs):
        rs = 4
        # Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = cross_validation.train_test_split(snpdatakrr, snptargetkrr,
        #                                          test_size=0.50, random_state=rs)
        Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = train_test_split(snpdatakrr, snptargetkrr,
                                                       test_size=0.50, random_state=rs)
        clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
        clfkrr.fit(Xkrr_train, ykrr_train)
        krrte_pred = clfkrr.predict(snpdatakrr)
        krrtr_pred = clfkrr.predict(Xkrr_train)
        tr_error = clfkrr.score(Xkrr_train, ykrr_train)
        te_error = clfkrr.score(snpdatakrr, y2_target)
        te_preds.append(krrte_pred)
        tr_preds.append(krrtr_pred)
    avg_te_pred = np.mean(te_preds,axis=0)
    avg_tr_pred = np.mean(tr_preds,axis=0)
    return avg_te_pred


def scaledata_xy(X2,y2_target):
    """
       No split & scale X and y_target data
    """
    y2 = y2_target
    #X2 = X2.drop('p_tanimoto',1)
    data2 = X2.copy()
    target2 = y2.copy()
    # convert pandas to numpy array:
    # nptarget2 = target2.as_matrix() #
    nptarget2 = target2.to_numpy()
    nptarget2 = np.ravel(nptarget2)
    nptarget2 = np.ndarray.flatten(nptarget2)
    #npdata2 = data2.as_matrix()
    if type(data2) !=  type(np.arange(2)):
        npdata2 = data2.to_numpy()
    else:
        npdata2 = data2
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
    # nptarget2 = target2.as_matrix();  #as_matrix() doesn't work on 2.17.2020
    nptarget2 = target2.to_numpy();
    nptarget2 = np.ravel(nptarget2);
    nptarget2 = np.ndarray.flatten(nptarget2)
    #npdata2 = data2.as_matrix()   #as_matrix() doesn't work on 2.17.2020
    if type(data2) !=  type(np.arange(2)):
        #print(type(npdata2))
        npdata2 = data2.to_numpy()
    else:
        npdata2 = data2
    # DO NOT SCALE TARGET DATA
    #snptarget2 = preprocessing.scale(nptarget2)
    snptarget2 = nptarget2
    snpdata2 = preprocessing.scale(npdata2)
    #X2scale = pd.DataFrame(snpdata2)
    #X2_train, X2_test, y2_train, y2_test = cross_validation.train_test_split(snpdata2, snptarget2,
    #                                                    test_size=split, random_state=rstate)
    X2_train, X2_test, y2_train, y2_test = train_test_split(snpdata2, snptarget2,
                                                         test_size=split, random_state=rstate)
    return X2_train, X2_test, y2_train, y2_test
