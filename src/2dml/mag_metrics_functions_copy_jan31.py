#
# mag_metrics_functions.py
# Modified for python3.6 version
#
# SOME CODES IN NOTEBOOK found Jab 31 which are different from codes where
# Create new versino of this file wiht codes form notebook only...

from math import floor, ceil
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

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
from alloy_functions import *



def get_atoms_info(df, df_elements):
    """
        calculate list of sum of energies of constituent atoms of compounds
    """
    elems_list_set = []
    frac_dict_list = []
    for ith, cmpd in df['formula'].iteritems():
        #if ith < 3:
        #print('\n')
        #print(cmpd)
        mp_cmpd = mp.Composition(cmpd)
        num_atoms = mp_cmpd.num_atoms
        #print(num_atoms)
        elems = mp_cmpd.elements
        elems_list = []
        frac_list = []
        fraction_dict = []
        energy_dict = []
        for el in elems:
            #print(el)
            elems_list.append(el)
            elem_frac = mp_cmpd.get_atomic_fraction(el)
            frac_list.append(elem_frac)
            elem_frac_dict = {el : elem_frac}
            fraction_dict.append(elem_frac_dict)
        frac_list = np.asarray(frac_list)
        # collect elements info
        elems_list_set.append(elems_list)
        frac_dict_list.append(fraction_dict)
    return elems_list_set, frac_dict_list


def energyplot(spinMatrix_dif, TMlist,B_atom_pair, title, cmaplabel,vmin,vmax,range=True):
    """
        constructe 2D energy difference plot using Matrix input
    """
    current_cmap = matplotlib.cm.get_cmap(name=cmaplabel)
    current_cmap.set_bad(color='grey')
    # masked_array=np.ma.masked_where(a==-999, a)
    # cmap = matplotlib.cm.jet
    # cmap.set_bad('w',1.)
    if range == True:
        plt.imshow(spinMatrix_dif, cmap=current_cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(spinMatrix_dif, cmap=current_cmap)
    #plt.colorbar(label='spin state $\Delta$ E')
    Batoms = ['_'.join(x) for x in B_atom_pair]
    x = np.arange(spinMatrix_dif.shape[1])
    y = np.arange(spinMatrix_dif.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.title(title,fontsize=15)
    plt.grid(False)
    plt.xticks(x, xlabels,rotation='vertical', fontsize=20)
    plt.yticks(y, ylabels,rotation='horizontal', fontsize=20)
    #plt.show()
    return


def spinMatrixGen(df_counts,TMlist,B_atom_pair,descriptor):
    """
        generates EnergyMAtrix from B list, TM list
        constrain for Te containing, or Se containing atoms etc prior
    """
    df_counts1 = df_counts.copy(deep=True)
    df_counts1 = df_counts1.reset_index()
    spinMatrix = np.ones((len(TMlist),len(B_atom_pair)))*3.141
    #print(TMlist)
    #print(B_atom_pair)
    for i,cmpd in enumerate(df_counts1['formula'][:]):
        #print(cmpd)
        for bth, b in enumerate(B_atom_pair):
            for tmth, tm in enumerate(TMlist):
                #print(b, tm)
                if tm == 'Cr': #special case when Cr is tm. always will find it ipresent...
                    TMtrue = df_counts1.loc[i,'Cr'] == 2.0
                    #print(TMtrue)
                    Btrue = Bexists(i, b, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_spinstate = df_counts1['spin_state'][i]
                        TMB_spinstate = df_counts1[descriptor][i]
                        #print(tm, b, TMB_spinstate)
                        if pd.isnull(TMB_spinstate):
                            spinMatrix[tmth,bth] = np.nan
                        else:
                            spinMatrix[tmth,bth] = TMB_spinstate
                else:
                    Btrue = Bexists(i, b, df_counts1)
                    TMtrue = TMexists(i, tm, df_counts1)
                    #print('got here', Btrue, TMtrue)
                    if Btrue and TMtrue:
                        #TMB_spinstate = df_counts1['spin_state'][i]
                        TMB_spinstate = df_counts1[descriptor][i]
                        if pd.isnull(TMB_spinstate):
                            #print(tm, b, TMB_spinstate)
                            spinMatrix[tmth,bth] = np.nan
                        else:
                            #print(tm, b, TMB_spinstate)
                            spinMatrix[tmth,bth] = TMB_spinstate

    spinMatrix_nna = spinMatrix.copy()
    spinMatrix_nna[np.isnan(spinMatrix_nna)] = np.nan #-1.0
    return spinMatrix_nna


def is_num(aa):
    try:
        float(aa)
        return True
    except ValueError:
        return False


class magnetization:

    def __init__(self, stem, target):
        self.stem = stem
        self.target = target
        ext = os.path.join(stem, target)
        self.ext = ext
        outs = os.listdir(ext)
        outs = [x for x in outs if '.' not in x]
        self.outs = outs
        formula = [x.split('_')[0] for x in self.outs]
        self.formula = formula


    def gen_formula(self):
        """
            generate formula list
        """
        formula = [x.split('_')[0] for x in self.outs]
        return formula


    def pull_mag_string(self, outpath):
        """ generate magnetization(i) string from a file """
        line_list = []
        #print(outpath)
        with open(outpath) as f:
            lines = f.readlines()
            mag_x = []
            mag_y = []
            mag_z = []
            for ith, line in enumerate(lines):
                if 'magnetization (x)' in line:
                    mag_stringx = lines[ith:ith+16]
                    mag_x.append(mag_stringx)
                if 'magnetization (y)' in line:
                    mag_stringy = lines[ith:ith+16]
                    mag_y.append(mag_stringy)
                if 'magnetization (z)' in line:
                    mag_stringz = lines[ith:ith+16]
                    mag_z.append(mag_stringz)
        return mag_x, mag_y, mag_z


    def pull_mag_string_noso(self, outpath):
        """ generate magnetization(i) string from a file """
        line_list = []
        with open(outpath) as f:
            lines = f.readlines()
            mag_x = []
            for ith, line in enumerate(lines):
                if 'magnetization (x)' in line:
                    mag_stringx = lines[ith:ith+16]
                    mag_x.append(mag_stringx)
        return mag_x




    #parse mag string info
    def parse_mag_string(self, mag_x):
        """
            get magnetization(x) portion of OUTCAR in string format and convert to array
        """
        if 'f' in mag_x[2]: #look for f string in appropriate line
            f_orbital = True
        else:
            f_orbital = False
            #print(mag_x[4:14])
        sites_str = mag_x[4:14]
        tot_str = mag_x[15:16]
        #print('sites_str',sites_str)
        #print('tot_str',tot_str)
        sites_str = [x.replace('\n','') for x in sites_str]
        sites_str = [x.split(' ') for x in sites_str]
        mod_sites_str = []
        for sites in sites_str:
            sites = [x for x in sites if x != '']
            mod_sites_str.append(sites)
        sites_str =  mod_sites_str
        all_digits=[]
        for j in np.arange(len(sites_str)):
            digits = [is_num(x) for x in sites_str[j]]
            all_digits.extend(digits)
        all_digits = np.asarray(all_digits)
        if all_digits.all():
            mag = np.asfarray(sites_str)
            if mag.shape[1] <= 5:
                mag = np.zeros((10,6))
                mag_temp = np.asfarray(sites_str)
                mag[:,:4] = mag_temp[:,:4]
                mag[:,5] = mag_temp[:,4]
            #print(mag.shape)
        else:
            #print("error in OUTCAR")
            if f_orbital == False:
                mag = np.zeros((10,6))*np.nan
            else:
                mag = np.zeros((10,6))*np.nan
        return mag


    def is_forbital(self, mag_x):
        """
            get magnetization(x) portion of OUTCAR in string format and convert to array
        """
        #print('magx',mag_x[2],'\n')
        if 'f' in mag_x[2]: #look for f string in appropriate line
            f_orbital = True
        else:
            f_orbital = False
        return f_orbital


    def get_relaxed_info(self, magarray_x_set, verbose):
        """ pick up relaxed info from magnetization data"""
        # extract magnetization results for relaxed structures
        mag_x_final = []
        #print('magarray_x_set',magarray_x_set)
        for ith, x in enumerate(magarray_x_set):
            if len(x) == 0:
                if verbose == True:
                    print(outs[ith], ' has no magnetic info')
                mag_x_final.append(np.nan)
            else:
                relaxed = x[-1]
                mag_x_final.append(relaxed)
        return mag_x_final


    def get_magdata(self, outs):
        """
            pull magnetization(i) strings and convert to arrays
        """
        magarray_x_set = []
        magarray_y_set = []
        magarray_z_set = []
        for ith, out in enumerate(outs):
            ext = self.ext
            outs = self.outs
            outpath = os.path.join(ext,out)
            mag_x, mag_y, mag_z = self.pull_mag_string(outpath)
            magarrays_x = [self.parse_mag_string(x) for x in mag_x]
            magarrays_y = [self.parse_mag_string(x) for x in mag_y]
            magarrays_z = [self.parse_mag_string(x) for x in mag_z]
            if len(mag_x) == 0:
                forbit = False
            else:
                if len(mag_x[0]) > 0:
                    forbit = self.is_forbital(mag_x[0])
                else:
                    forbit = False

            # collect only final magnetization (x) data or all right now?
            magarray_x_set.append(magarrays_x)
            magarray_y_set.append(magarrays_y)
            magarray_z_set.append(magarrays_z)
        return magarray_x_set, magarray_y_set, magarray_z_set, forbit


    def get_magdata_noso(self,outs):
        """
            pull magnetization(i) strings and convert to arrays
            no spin orbit
        """
        magarray_x_set = []
        for ith, out in enumerate(outs):
            #print('out', out)
            #ext = self.ext
            #outs = self.outs
            stem = self.stem
            target = self.target
            path = os.path.join(stem, target)
            outpath = os.path.join(path,out)
            mag_x = self.pull_mag_string_noso(outpath)
            magarrays_x = [self.parse_mag_string(x) for x in mag_x]
            if len(mag_x) > 0:  # sometimes OUTCAR file has missing data
                forbit = self.is_forbital(mag_x[0])
            else:
                forbit = False
            magarray_x_set.append(magarrays_x)
        return magarray_x_set, forbit


    def mag_tots_gen(self, mag_z_final, forbit):
        """
            generate mag_tot and mag_d given mag_i_final array
        """
        # extract elements of magnetization info
        mag_tots = []
        mag_ds = []
        magshape = 0
        for mag in mag_z_final:
            if (np.isnan(mag).any()) == False:
                magshape = (mag.shape)
                #if forbit == False:
                mag_tot = (mag[:,5])  #get sum of spdf mag contricution for all 1 through 10 sites.
                mag_d = np.sum(mag[:,3:5],axis=1)  # sum up d and f orbitals
                #else:
                #    mag_tot = (mag[:,5])
                #    #mag_d = (mag[:,4])
                mag_tots.append(mag_tot)
                mag_ds.append(mag_ds)
            else:
                if magshape != 0:
                    nanarray = np.zeros(magshape[0])*np.nan
                else:
                    nanarray = np.zeros((10,))*np.nan
                mag_tots.append(nanarray)
                mag_ds.append(nanarray)
        return mag_tots, mag_ds



    def gen_localmag_df(self, stem, target):
        """
            Create datafraome containing local magnetic moments given target folder
            input: target
            output: df_mag
        """
        # stem = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/local_mu/'
        ext = stem + target
        outs = os.listdir(ext)
        outs = [x for x in outs if '.' not in x]

        #maginfo = magnetization(ext)
        #outs = maginfo.outs
        magarray_x_set, magarray_y_set, magarray_z_set, forbit = self.get_magdata(outs)
        mag_x_final = self.get_relaxed_info(magarray_x_set,False)
        mag_y_final = self.get_relaxed_info(magarray_y_set,False)
        mag_z_final = self.get_relaxed_info(magarray_z_set,False)

        formula = self.gen_formula()
        #444
        mag_tots, mag_ds = self.mag_tots_gen(mag_z_final, forbit)
        mag_x_tots, mag_x_ds = self.mag_tots_gen(mag_x_final, forbit)
        mag_y_tots, mag_y_ds = self.mag_tots_gen(mag_y_final, forbit)

        # collect mag_tot for all ten sites into mag_tots_arr
        mag_tots_arr = np.vstack(mag_tots)
        mag_x_tots_arr = np.vstack(mag_x_tots)
        mag_y_tots_arr = np.vstack(mag_y_tots)
        #print('mag_tots_arr', mag_tots_arr)
        ### Create dataframe
        #print('mag_tots_arr[:,0]',mag_tots_arr[:,0])
        #         mz_Cr = mag_tots_arr[:,0]
        #         mz_TM = mag_tots_arr[:,1]
        #         mx_Cr = mag_x_tots_arr[:,0]
        #         mx_TM = mag_x_tots_arr[:,1]
        #         my_Cr = mag_y_tots_arr[:,0]
        #         my_TM = mag_y_tots_arr[:,1]

        mz_CrTM = mag_tots_arr[:,:2]
        mx_CrTM = mag_x_tots_arr[:,:2]
        my_CrTM = mag_y_tots_arr[:,:2]
        TM_ions = []
        #print(ext)
        for out in outs:
            #print(out)
            path = ext + '/' + out
            TM_ion = self.pull_ion_order(path)
            TM_index = TM_ion[1]
            TM_ions.append(TM_index)
        TM_ions = np.asarray(TM_ions)
        #print('TM_ions shape',TM_ions.shape)
        #print('mz_CrTM[TM_ions] sha', mag_tots_arr[:,:2].shape, mz_CrTM[TM_ions].shape, mz_CrTM[TM_ions])
        #print('TM_ions', TM_ions)

        mz_Cr = reorder(mz_CrTM, TM_ions)[:,0]
        mz_TM = reorder(mz_CrTM, TM_ions)[:,1]
        mx_Cr = reorder(mx_CrTM, TM_ions)[:,0]
        mx_TM = reorder(mx_CrTM, TM_ions)[:,1]
        my_Cr = reorder(my_CrTM, TM_ions)[:,0]
        my_TM = reorder(my_CrTM, TM_ions)[:,1]
        #         mz_Cr  = mz_CrTM[:,TM_ions]#[:,0]
        #         mz_TM  = mz_CrTM[:,TM_ions][:,1]
        #         mx_Cr  = mx_CrTM[:,TM_ions][:,0]
        #         mx_TM  = mx_CrTM[:,TM_ions][:,1]
        #         my_Cr  = my_CrTM[:,TM_ions][:,0]
        #         my_TM  = my_CrTM[:,TM_ions][:,1]
        #print('mz_Cr.shape',mz_Cr.shape)
        #print('mz_Cr', mz_Cr)

        df_mag = pd.DataFrame()
        df_mag['formula'] = formula
        df_mag['mz_Cr'] = mz_Cr
        df_mag['mz_TM'] = mz_TM
        df_mag['mx_Cr'] = mx_Cr
        df_mag['mx_TM'] = mx_TM
        df_mag['my_Cr'] = my_Cr
        df_mag['my_TM'] = my_TM
        return df_mag


    def gen_localmag_df_noso(self, target):
        """
            Create datafraome containing local magnetic moments given target folder
            No spin-orbit coupling output
        """
        ext = self.ext  #stem + target
        outs = self.outs
        magarray_x_set, forbit = self.get_magdata_noso(outs)
        mag_x_final = self.get_relaxed_info(magarray_x_set,False)
        formula = self.gen_formula()
        mag_tots, mag_ds = self.mag_tots_gen(mag_x_final, forbit)

        # collect mag_tot for all ten sites into mag_tots_arr
        mag_tots_arr = np.vstack(mag_tots)

        ### Create dataframe
        mx_Cr = mag_tots_arr[:,0]
        mx_TM = mag_tots_arr[:,1]

        df_mag_so = pd.DataFrame()
        df_mag_so['formula'] = formula
        df_mag_so['mx_Cr'] = mx_Cr
        df_mag_so['mx_TM'] = mx_TM
        return df_mag_so


    def pull_ion_order(self, outpath):
        """
            generate ion order string from a file
            input: outpath
            output: list of ions from OUTCAR
        """
        with open(outpath) as f:
            lines = f.readlines()
            ions = []
            for ith, line in enumerate(lines):
                if 'TITEL ' in line:
                    ion = line.split(' ')[7]
                    ion = ion.split('_')
                    if len(ion) > 1:
                        ion = ion[0]
                    else:
                        ion = ion[0]
                    ions.append(ion)
        ion_index = np.argwhere(np.asarray(ions) == 'Cr')[0][0]
        #print(ion_index)
        if ion_index == 0:
            ion_index = np.asarray([0,1])
        else:
            ion_index = np.asarray([1,0])
        #print(ion_index)
        return ions, ion_index




def reorder(original, order):
    """
        input: original, order
        output: sorted array given by order
        used to correct placement of ion. That is, Cr vs TM
    """
    sorted = np.empty(original.shape)
    for ith, i in enumerate(order):
        sorted[ith,:] = original[ith,i]
    return sorted



def gen_tally(df_mag, main_dir, recalculate):
    """
        creates df_spin_counts from df_mag
        input: df_mag, recalculate
        output: df_spin_counts
    """
    # recalculate = True
    df_elements = get_unique_elem_info(main_dir, df_mag, recalculate=recalculate)
    #df_elements.head()

    elems_list_set, frac_dict_list = get_atoms_info(df_mag, df_elements)
    df_mag["elem_list"] = elems_list_set
    df_mag["elem_frac"] = frac_dict_list
    elemlist = "elem_list"
    atom_label_list_spin, atom_count_list_spin = get_atom_counts(df_mag, df_elements, elemlist)
    df_spin_counts_Te = df_mag.copy(deep = True)
    for ith, atom_label in enumerate(atom_label_list_spin):
        df_spin_counts_Te[atom_label] = atom_count_list_spin[ith]
    return df_spin_counts_Te




def is_Te_vec(edf):
    """
        get vec of is Te?
    """
    is_Te = []
    for entry in edf['elem_list_spin'].values:
        entry_list = [x.as_dict()['element'] for x in entry]
        if 'Te' in (entry_list):
            is_Te.append(1)
        else:
            is_Te.append(0)
    is_Te = np.asarray(is_Te)
    return is_Te



def sort_df(df_fm_afm, edf_sub):
    """
        Sort data based on df_fm_afm.
        input: - df_fm_afm and edf for sorting
        returns: - dataframe with sorted data
    """
    # sort contcar data to match file names in df_spin_so
    sortdex = []
    formula_set = edf_sub["formula_Hf"].values
    for ith, item in enumerate(df_fm_afm['formula_fm']):
        index = np.where(np.asarray(formula_set) == item)[0][0]
        sortdex.append(index)
    formula_set = np.asarray(formula_set)
    formula_sorted = formula_set[sortdex]
    #print(sortdex)
    #print(formula_sorted)
    edf_sorted = edf_sub.iloc[sortdex,:].copy()
    edf_sorted = edf_sorted.drop(columns=['level_0', 'index'])
    edf_sorted = edf_sorted.reset_index()
    return edf_sorted



def get_df_optima_local(df_master):
    """create df with nin E configurations"""
    spinlabel = ['spin_so','afm']
    minE = []
    optMag_Cr = []
    optMag_TM = []
    spinlabels = []
    spindex = []
    for c in df_master[:].iterrows():
        rownum = c[0]
        energies = (c[1][['cohesive_spin_so','cohesive_afm']])
        magmoms_Cr = (c[1][['mz_Cr_fm','mz_Cr_afm']])
        magmoms_TM = (c[1][['mz_TM_fm','mz_TM_afm']])
        magmoms_Cr = np.abs(magmoms_Cr)
        magmoms_TM = np.abs(magmoms_TM)
        mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
        optimum_mag_Cr = magmoms_Cr[mindex_energy]
        optimum_mag_TM = magmoms_TM[mindex_energy]
        optMag_Cr.append(optimum_mag_Cr)
        optMag_TM.append(optimum_mag_TM)
        minE.append(np.min(energies))
        spindex.append(mindex_energy)
        spinlabels.append(spinlabel[mindex_energy])
        #print(mindex_energy,optimum_mag)
    df_optima = pd.DataFrame()
    df_optima['formula'] = df_master['formula_Hf'].values
    df_optima['minE'] = minE
    df_optima['optMag_Cr'] = optMag_Cr
    df_optima['optMag_TM'] = optMag_TM
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    return df_optima


def get_df_optima_local_noso(df_master):
    """create df with nin E configurations"""
    spinlabel = ['spin','afm_noso']
    minE = []
    optMag_Cr = []
    optMag_TM = []
    spinlabels = []
    spindex = []
    for c in df_master[:].iterrows():
        rownum = c[0]
        energies = (c[1][['cohesive_spin','cohesive_afm_noso']])
        magmoms_Cr = (c[1][['mx_Cr_fm_noso','mx_Cr_afm_noso']])
        magmoms_TM = (c[1][['mx_TM_fm_noso','mx_TM_afm_noso']])
        magmoms_Cr = np.abs(magmoms_Cr)
        magmoms_TM = np.abs(magmoms_TM)
        mindex_energy = np.argwhere(energies == np.min(energies))[0][0]
        optimum_mag_Cr = magmoms_Cr[mindex_energy]
        optimum_mag_TM = magmoms_TM[mindex_energy]
        optMag_Cr.append(optimum_mag_Cr)
        optMag_TM.append(optimum_mag_TM)
        minE.append(np.min(energies))
        spindex.append(mindex_energy)
        spinlabels.append(spinlabel[mindex_energy])
        #print(mindex_energy,optimum_mag)
    df_optima = pd.DataFrame()
    df_optima['formula'] = df_master['formula_Hf'].values
    df_optima['minE'] = minE
    df_optima['optMag_Cr'] = optMag_Cr
    df_optima['optMag_TM'] = optMag_TM
    df_optima['spinlabels'] = spinlabels
    df_optima['spindex'] = spindex
    return df_optima
