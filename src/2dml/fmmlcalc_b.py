##
## Set of functios to be used with FMML.ipynb
##

# creates: band_alignment.png
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

#from pymatgen.matproj.rest import MPRester
from pymatgen.ext.matproj import MPRester

#This initializes the REST adaptor. Put your own API key in.
m = MPRester("RK5GrTk1anSOmgAU")
import pymatgen as mg
from mendeleev import element
from fractions import Fraction

import itertools
#from pymatgen.analysis.structure_analyzer import VoronoiCoordFinder

def deuterium(str):
    """ Converts deuterium chemical symbol, D used in database to H.
        Consider way to keep track of atomic number
    """
    if 'D' in str:
        if 'Dy' in str:
            #Need to account for elements iwth D byt not deuterium 'Dy'
            return str
        else:
            #print re.sub(u'D(=?\d)', u'H(=?\d)', str)
            re_object = re.findall(r"D+\d", str)
            #print 'ob',  re_object
            object_list = list(re_object[0])
            for ith, i in enumerate(object_list):
                #print ith, i
                if i == 'D':
                    object_list[ith] = 'H'
            new_val = ''.join(object_list)
            str = re.sub('D(=?\d)',new_val,str)
            #print 'mew value for a', str
            return str
    else:
        return str


def make_rational(formula):
    """returns the scaling ratio for components in a formula"""
    comp = mg.Composition(formula)
    val = comp.values()
    new_val, scaling = rationalize(val)
    return scaling


def rationalize(arr):
    """Converts ratio of components to integer numbers
       Return the corrected list of numbers and the scaling factor
    """
    denom_max=0
    arr = np.asfarray(arr)
    for ith in arr:
        #r = ith.as_integer_ratio() #old codes
        r = Fraction(ith).limit_denominator(1000)
        #print r
        denom = r.denominator
        #print denom
        if denom_max < denom:
            denom_max = denom
        #if denom_max < r[1]: #part of old codes
        #    denom_max = r[1]
    arr = arr*denom_max
    arr = list(arr)
    return arr, denom_max



def makepretty(mystr,rescale=False):
    """
        Converts formula input into a form materials project APS can understand
        * 1.22.2017 incorporate pymatgen get_reduced_formula_and_factor() to eliminate parens
        * 2.6.2017 updated code to correct issue with finding fraction respresentatin from decimal
        * Convert 2-x to 1.9 and x to 0.1 to deal with doping.. Better way? CHeck this performance?
        """
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
        #
        #if there is some decimals that doesnt result in good
        #rescaling, ie scaling >>10, then jsut round the numbers.
        if scaling < 10:
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



def dehydrate(fname):
    """
        deals with hydration in formulae of magnetic susceptibiltiy datasets
        retuns string that makepretty() function can process.
    """
    print('fname',fname)
    #fname = fname.encode('utf-8')
    #print('dehydr', type(fname), fname)
    fname_list = fname.split('\xc2\xb7')
    if len(fname_list) > 1:
        main = fname_list[0]
        stem = fname_list[1]
        #print main, stem
        mystr = re.findall(r'(\d+)[A-Z]',stem)#.group(0)
        water_c = mystr[0]
        new_stem = '(H2O)' + str(water_c)
    else:
        main = fname_list[0]
        water_c = 1
        new_stem = ''
    str_update = main + new_stem
    return str_update



def makepretty2(mystr,rescale=False):
    """
        Converts formula input into a form materials project APS can understand
        * 1.22.2017 incorporate pymatgen get_reduced_formula_and_factor() to eliminate parens
        * 2.6.2017 updated code to correct issue with finding fraction respresentatin from decimal
        * Convert 2-x to 1.9 and x to 0.1 to deal with doping.. Better way? CHeck this performance?
        * Updated Mar.4.2014 to handle magnetic susceptibilities dataset and hydradtion '/dotH20'
    """
    #print 'mnake pretty2', mystr
    mystr = re.sub(r'\xa0','',mystr)
    #
    #mystr = dehydrate(mystr)
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
    mystr = mystr.replace(' ','')
    #mystr = unicode(mystr)
    #mystr = mystr.decode('utf8')
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
        #
        # if there is some decimals that doesnt result in good
        # rescaling, ie scaling >>10, then jsut round the numbers.
        if scaling < 10:
            for elem in liststr:
                counter = counter + 1
                u = unicode(elem.replace('.',''))
                isnum = u.isnumeric()
                if isnum == True:
                    newelem = float(elem)
                    newelem = newelem*scaling
                    newelem = str(int(newelem))
                    # newelem = str(int(round(newelem)))
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




def getM(posmatrix,atomN):
    """
    Creates Coulomb matrix from positions and atomic numbers
    """
    n=len(posmatrix)
    M=np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            if i==j:
                Zi = atomN[i]
                M[i,j]=0.5*Zi**2.4
            else:
                Zi=atomN[i]
                Zj=atomN[j]
                #print 'Zi', Zi, 'Zj', Zj
                rij = (np.sum((posmatrix[i,:] - posmatrix[j,:])**2.0))**0.5
                #rij = (np.sum(np.abs(posmatrix[i,:] - posmatrix[j,:])))
                M[i,j]= Zi*Zj/(rij)
    return M




def Colkernel(M):
    """
    Implement the Coulomb Kernel
    - updated square of eval terms 2.13.2017
    """
    eval1 = np.linalg.eigvals(M)
    evalprime = np.zeros(M.shape)[0] #don't get nxm matrix....
    dMM =(np.sum(np.abs((eval1-evalprime)**2.0)))**0.5
    return dMM


def d_gen(poss,dfname):
    dlist=[]
    for i in dfname.index:
        if i <= len(dfname):
            M1 = getM(poss[0],atomNs[0])
            M2 = getM(poss[i],atomNs[i])
            d = Colkernel(M1,M2)
            dlist.append(d)
    return dlist



def veclength(vec):
    len_sq = []
    for dim in vec:
        len_sq.append(dim**2.0)
        #print 'len', len_sq
    veclen = np.sqrt(np.sum(len_sq))
    return veclen

def get_abc(strucdata_i):
    length_angle = strucdata_i.lattice.lengths_and_angles
    #print 'length angle: ', length_angle[0]
    lengths = length_angle[0]
    a=lengths[0]; b=lengths[1]; c=lengths[2]
    r_ab = a/b
    r_ac = a/c
    ratios = [r_ab,r_ac]
    ratios = np.asarray(ratios)
    np.reshape(ratios,(1,len(ratios)))
    return ratios



def max_atomic_mass(fdata):
    """works with getmpidata and call to m.get_data(formula)
       to extract a list of elements with their atomic masses.
       Extract the heaviest element? The average of all elements?
    """
    compound_Z = []
    for i, elem in enumerate(fdata):
        label_Z = []
        compound = elem['elements']
        for atom in compound:
            mg_atom = mg.Element(atom)
            Z = mg_atom.atomic_mass
            label_Z.append([atom,Z])
        #print '#', i, label_Z
        compound_Z.append(label_Z)
    #print 'answer', compound_Z
    return compound_Z



def max_difference(descrip_set):
    """ calculates the max differnce"""
    w_dif_set = []
    #print descrip_set
    for descrip in descrip_set:
        #print 'ee', descrip
        #print 'in None?', isnone
        if not type([]) == type(descrip):
            #print 'got here'
            w_dif_set.append(np.nan)
        else:
            isnone = None in descrip
            if isnone:
                w_dif_set.append(np.nan)
            else:
                descrip = np.asarray(descrip)
                max = np.max(descrip)
                min = np.min(descrip)
                #print max
                #print min
                dif = max - min
                w_dif_set.append(dif)
    return w_dif_set




def weighted_avg(descrip_set, w_set):
    """calculates the weighted average"""
    w_avg_set = []
    for jth, descrip in enumerate(descrip_set):
        #print 'length',len(descrip_set)
        #print 'tp', jth, descrip
        #print 'wset', w_set
        w = w_set[jth]
        avg = []
        if not type([]) == type(descrip):
            w_avg_set.append(np.nan)
        else:
            if None in descrip:
                w_avg_set.append(np.nan)
            else:
                if np.isnan(descrip).any():
                    w_avg_set.append(np.nan)
                    continue
                for ith, item in enumerate(descrip):
                    #print 'in weighted avg loop'
                    #print (ith)
                    #print (item)
                    #print 'end loop'
                    avg.append(w[ith]*item)
                npavg = np.asarray(avg)
                w_avg = np.sum(npavg)
                w_avg_set.append(w_avg)
    return w_avg_set





def getMGdata(df):
    """Gethers data gathered from pymatgen for all formulae in a dataframe"""
    #row = next(df.iterrows())[1]
    Z_collection = []
    ox_col = []
    ionic_col = []
    pretty_name = []
    doping_names = []
    #print 'got here'
    for ith, fname in df['formula'].iteritems():
        #print fname
        dope_name = makepretty_doping(fname) #call makepretty_doping before changing contents of fname
        fname = makepretty2(fname) #need to eliminate white spaces and fractional constituents
        pretty_name.append(fname)
        atom_list, Z_list, ionic_list, ox_list = get_atomic_info(fname)
        Z_collection.append(Z_list)
        ox_col.append(ox_list)
        ionic_col.append(ionic_list)
        doping_names.append(dope_name)
    Z_collection = np.asarray(Z_collection)
    ox_col = np.asarray(ox_col)
    ionic_col = np.asarray(ionic_col)
    pretty_name = np.asarray(pretty_name)
    return Z_collection, ox_col, ionic_col, pretty_name, doping_names






def avg_Z(df):
    """
      Examines df['zdata'] and gets mean of Z as proxy for determining
      impact of spin-orbit coupling
    """
    avgZ = []
    for ith, zval in df['zdata'].iteritems():
        #print zval
        if isinstance(zval, list):
            avg = np.mean(zval)
            #print avg
        else:
            avg = np.nan
        avgZ.append(avg)
    return avgZ


def get_atomic_info(fname):
    """works with addMGdata and call to m.get_data(formula)
       to extract a list of elements with their atomic masses.
       Extract the heaviest element? The average of all elements?
       - include average ionic radius
       - include average common oxidation states
       - need to account for case where element does not exist. see if statement
    """
    Z_list = []
    atom_list = []
    ox_list = []
    ionic_list = []
    fdata = m.get_data(fname)
    if fdata == []:
        return [np.nan, np.nan, np.nan, np.nan]
    mpielements = fdata[0]['elements']
    for elem in mpielements:
        mg_atom = mg.Element(elem)
        #print mg_atom
        Z = mg_atom.atomic_mass
        atom_list.append(elem)
        Z_list.append(Z)
        ionic_r = mg_atom.average_ionic_radius
        #print 'ionic', ionic_r
        ox_state = mg_atom.common_oxidation_states
        avg_ox_state = np.mean(ox_state)
        #print avg_ox_state
        ionic_list.append(ionic_r)
        ox_list.append(avg_ox_state)
    return atom_list, Z_list, ionic_list, ox_list



def shells(shell):
    """ Takes the last subshell given by mendeleev package and returns
        The sum of 2p and d electrons given by the formula.
        What about # electrons within the unit cell?
        - updated to get only 'p' electrons
    """
    cmpd_p = []
    cmpd_d = []
    cmpd_f = []
    cmpd_mean_p = []
    cmpd_mean_d = []
    cmpd_mean_f = []
    cmpd_dif_p = []
    cmpd_dif_d = []
    cmpd_dif_f = []
    for l in shell:
        N = len(l)
        num_p = [0]
        num_d = [0]
        num_f = [0]
        for ith in l:
            #print ith[0], ith[1]
            if ith[0][1] == 'p': #ith[0] ==(2,u'p'):
                num_p.append(ith[1])
            if ith[0][1] == 'd':
                num_d.append(ith[1])
            if ith[0][1] == 'f':
                num_f.append(ith[1])
        num_p = np.asarray(num_p)
        num_d = np.asarray(num_d)
        num_f = np.asarray(num_f)
        #print 'nump', num_p
        #print 'numd', num_d
        max_dif_p = np.max(num_p) - np.min(num_p)
        max_dif_d = np.max(num_d) - np.min(num_d)
        max_dif_f = np.max(num_f) - np.min(num_f)
        sum_p = np.sum(num_p)
        sum_d = np.sum(num_d)
        sum_f = np.sum(num_f)
        mean_p = sum_p/N
        mean_d = sum_d/N
        mean_f = sum_f/N
        cmpd_p.append(sum_p)
        cmpd_d.append(sum_d)
        cmpd_f.append(sum_f)
        cmpd_mean_p.append(mean_p)
        cmpd_mean_d.append(mean_d)
        cmpd_mean_f.append(mean_f)
        cmpd_dif_p.append(max_dif_p)
        cmpd_dif_d.append(max_dif_d)
        cmpd_dif_f.append(max_dif_f)
    return (cmpd_p, cmpd_d, cmpd_f, cmpd_mean_p, cmpd_mean_d,
            cmpd_mean_f, cmpd_dif_p, cmpd_dif_d, cmpd_dif_f)


from scipy.stats import skew

def shells_stats(shell):
    """
	Takes the last subshell given by mendeleev package and returns
        The sum of 2p and d electrons given by the formula.
        What about # electrons within the unit cell?
        - updated to get only 'p' electrons
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
        sigma_p = np.std(num_p)
        sigma_d = np.std(num_d)
        sigma_f = np.std(num_f)
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


def get_ionization(ionenergies):
    """ get first three ionization energies. OR first if Hydrogen"""
    entry_list = []
    dif_list = []
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
        dif_ion = np.max(ion_list) - np.min(ion_list)
        sum_ion = np.sum(ion_list)
        mean_ion = np.mean(ion_list)
        entry_list.append(ion_list)
        dif_list.append(dif_ion)
        sum_list.append(sum_ion)
        mean_list.append(mean_ion)

    return entry_list, dif_list, sum_list, mean_list


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



def get_FM(newdf):
    """ Use net magnetic moment to determine whether a material is FM or not.
        Report result using 1 or 0.
    """
    FM = []
    for mu in newdf['mnet']:
        #print mu
        if mu < 1.0:
            FM.append(0)
        else:
            FM.append(1)
    N_Tot = len(FM)*1.0
    N_FM = np.sum(FM)*1.0
    frac_FM = N_FM/N_Tot
    return FM, frac_FM


def get_mse(y_test,prediction):
    """
        Calculates the Mean Squared Error of test data and predictions
    """
    acc = np.mean((y_test-prediction)**2.0);
    return acc


#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Reds):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)###
#
#    if normalize:
##        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')#

#    print(cm)

#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, cm[i, j],
#                 horizontalalignment="center",
#                 color="grey" if cm[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="grey" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def get_eTot(numelectron):
    """ Gets the number fo electrons for each atomic specias in a compound and sums them up
        returning the total number of electrons for a compound.
    """
    sum_e_list = []
    for cmpd in numelectron:
        np_cmpd = np.asarray(cmpd)
        sum_e = np.sum(np_cmpd)
        sum_e_list.append(sum_e)
    sum_e_list = np.asfarray(sum_e_list)
    sum_e_list = 1.0*(sum_e_list)
    return sum_e_list




def get_min_dist(posmatrix):
    """
        Returns min distance in a position matrix representing
        the differences of the atomic positions
        NB: should implement this for MAGNETIC ATOMS only....
         - should not account for on-diagonal terms!!!!
    """
    dist_array = np.zeros((len(posmatrix),1))
    for i, pos in enumerate(posmatrix):
        if np.isnan(pos).any():
            min_dist = np.nan
        else:
            atomF = np.ones((pos.shape[0],1))
            M = posM(pos,atomF)
            index = np.where(M == M.min())
            #print pos
            #print index
            #index = np.argmin(M) # this only returns one value
            #if len(index) > 1:
            #   print 'more than one nearest neighbour'
            #min_dist = M.flat[index] # This .flat does not work in this case
            min_dist = M[index]
            avg_min_dist = np.mean(min_dist)
            #TwoD_index = np.unravel_index(index, M.shape)
        dist_array[i] = avg_min_dist
    return dist_array

#test = newdf['atom_pos']
#print test
#min_dist = get_min_dist(test)
#print min_dist.shape
#print min_dist


def get_avg_dist(posmatrix):
    """
        Returns min distance in a position matrix representing
        the differences of the atomic positions
        NB: should implement this for MAGNETIC ATOMS only....
         - should not account for on-diagonal terms!!!!
    """
    dist_array = np.zeros((len(posmatrix),1))
    for i, pos in enumerate(posmatrix):
        if np.isnan(pos).any():
            min_dist = np.nan
        else:
            atomF = np.ones((pos.shape[0],1))
            M = posM(pos,atomF)
            #index = np.where(M == M.min())
            #min_dist = M[index]
            #avg_min_dist = np.mean(min_dist)
            avg_dist = np.mean(M)
            #print avg_dist
        #dist_array[i] = avg_min_dist
    return avg_dist

#avg_distance = get_avg_dist(atom_coordinates[:4])

def orbital_gen(newdf2):
    #ADDED a lot of dummy variables for atoms where was no informatino.. try to inpute these values..
    """
       Generates columns with orbital radii using orbital_radii.csv
    """
    orbitals = pd.read_csv('orbital_radii.csv',header=0)
    orbit_df = pd.DataFrame()
    rs_df = []
    rp_df = []
    rd_df = []
    N = len(newdf2)
    #print N
    for ith, elem in enumerate(newdf2['prettyformula'][:]):
        #print elem
        atoms = mg.Composition(elem)
        rs_orbits = []
        rp_orbits = []
        rd_orbits = []
        for a in atoms:
            a = str(a)
            #if ith > 625:
            #    print a
            index = np.argwhere(a == orbitals.iloc[:,0])[0][0]
            rs = orbitals.iloc[index,1]
            rp = orbitals.iloc[index,2]
            rd = orbitals.iloc[index,3]
            rs_orbits.append(rs)
            rp_orbits.append(rp)
            rd_orbits.append(rd)
        rs_df.append(np.asarray(rs_orbits))
        rp_df.append(np.asarray(rp_orbits))
        rd_df.append(np.asarray(rd_orbits))
    orbit_df['rs'] = rs_df
    orbit_df['rp'] = rp_df
    orbit_df['rd'] = rd_df
    return orbit_df

def df_avg(descrip_set):
    """calculates the average"""
    avg_set = []
    for jth, descrip in enumerate(descrip_set):
        des = np.asarray(descrip)
        w_avg = np.mean(des)
        #print len(descrip_set)
        #print w_avg
        avg_set.append(w_avg)
    return avg_set




#max_df_diff(rs)



def posM(posmatrix,atomF):
    """
    - Creates atomic_position matrix from positions and atomic numbers
    - Could omit the second feature
    """
    n = len(posmatrix)
    #print 'n', n
    M = np.zeros((n,n))
    for i in np.arange(n):
        for j in np.arange(n):
            if i == j:
                Zi = atomF[i]
                M[i,j] = 50.0 #just make this very large to avound being picked up
            else:
                Zi = atomF[i]
                Zj = atomF[j]
                #print 'Zi', Zi, 'Zj', Zj
                #print posmatrix
                rij = (np.sum((posmatrix[i,:] - posmatrix[j,:])**2.0))**0.5
                #print 'rij', rij
                #rij = (np.sum(np.abs(posmatrix[i,:] - posmatrix[j,:])))
                #M[i,j]= Zi*Zj/(rij)
                M[i,j] = rij
    return M




def getbest(mpidata):
    """
    Get the most stable strcture. May not be a strucuture wit 0meV above Hull.
    Should find some good way of accounting for whether it's 0meV or just a
    small number or the lowest number
    Accounts for missing database entries
    """
    ll = len(mpidata)
    mpidata=np.asarray(mpidata)
    mymin = np.min(mpidata[:,0])
    index = np.where(mpidata[:,0]==mymin)
    #if mymin != 0:
        #print 'Ehull not zero, Ehull is: ', mymin
    #print index
    if len(index) == 0:
        print('not present in database')
    if len(index) > 1:
        print('more than one Ehull=0 (or minimum value)')
    return mpidata[index][0][1:], index[0][0]



def get_fingerprint(struct):
    """
       input: structure object from materials project database
       return: atomic_positions and fingerprint
    """
    Znum = struct.atomic_numbers
    Pos = struct.cart_coords
    M = getM(Pos,Znum)
    fp = Colkernel(M)
    return fp, Pos


def generate_feature(df):
    """ Parses unformatted susceptibilites data and
        returns two columns for formulas and susceptibilities
    """
    N = len(df)
    formula_list = []
    data_array = []
    #data_array = np.zeros((N,1)) #some entries are strings, eg Ferro.
    for i, ith in enumerate(df.iloc[:,0]):
        item = ith.split(' ')
        data = item[-2:]
        data = np.asarray(data)
        data_formula = data[0]
        data[1] = data[1].replace('+','')
        data_suscep = data[1]
        label = item[:-2]
        label = ' '.join(label)
        #data_suscep = np.float(data_suscep)
        #data_array[i] = data_suscep
        data_array.append(data_suscep)
        formula_list.append(data_formula)
    return formula_list, data_array


def remove_ferro(df2):
    """
        removes 'Ferro.' from susceptibility columns and casts strings to floats
    """
    ferro_index = np.argwhere(df2['suscep'] == 'Ferro.')
    ferro_index = np.ravel(ferro_index)
    #print 'here', df2.iloc[ferro_index,1]
    df2['suscep'][ferro_index] = 0.3e6  # gave warning!!!
    #df2.loc['suscep',ferro_index] = 1e6  # this gives error
    #print 'now', df2['suscep'][ferro_index]
    ##ferro_index1 = np.argwhere(df2['suscep'] == 'Ferro.')
    for ith, item in enumerate(df2['suscep']):
        if type(item) == str:
            newitem = item.replace('+','')
            df2['suscep'][ith] = newitem
    df2['suscep'] = df2['suscep'].astype(float)
    #df2values = df2['suscep'].values
    #df2log = np.log(df2values)
    return df2


def symmetrize(principal):
    """
       generalizes symmetrization of principal descriptors as decribed in
       'ML bandgaps of double perovskites' to beyond binary (AB) compounds.
       Here take a maximum ternary compound and implements symmetrization
       of contintuent parts
       Input: values to me symmetrized
       return: 8x1 matrix of symmetrized combinatinos of primary descriptors
               dataframe containing values from the matrix
    """
    maxN = 4
    df_features = np.zeros((len(principal),2*maxN))
    zero_matrix = np.zeros((1,maxN))
    ith_array = np.zeros((len(principal),maxN))
    for i, ith in enumerate(principal):
        #print i
        if np.isnan(ith).any():
            ith = np.nan
            ith_array[i,:] = ith
        else:
            ith = np.asarray(ith)
            ith_len = ith.shape[0]
            #print 'ith', ith
            dif_len = maxN - ith_len
            #print dif_len
            zero_append = np.zeros((1,dif_len))
            #print 'zer', zero_append
            ith_array[i,:] = np.concatenate((ith,zero_append[0]))
        mvals = [0,1]
        total_features = []
        for m in mvals:
            #m = 0
            operations = [[1,1,1,(-1)**m],[1,-1,1,(-1)**m],[1,1,-1,(-1)**m],[1,-1,-1,(-1)**m]]
            operations = np.matrix(operations)
            #print operations
            ith_sample = np.matrix(ith_array[i,:])
            #print ith_sample
            #print (ith_sample.T).shape
            #print operations.shape
            #sym_features =  np.dot(operations,ith_array[i,:])
            sym_features =  np.abs((operations*ith_sample.T))
            total_features.append(sym_features)
        total_features = np.asarray(total_features)
        total_features = np.ravel(total_features)
        df_features[i,:] = total_features
        return df_features


def orbital_gen(newdf2):
    #ADDED a lot of dummy variables for atoms where was no informatino.. try to inpute these values..
    """
       Generates columns with orbital radii using orbital_radii.csv
    """
    orbitals = pd.read_csv('orbital_radii.csv',header=0)
    orbit_df = pd.DataFrame()
    rs_df = []
    rp_df = []
    rd_df = []
    N = len(newdf2)
    #print N
    for ith, elem in enumerate(newdf2['prettyformula'][:]):
        #print elem
        atoms = mg.Composition(elem)
        rs_orbits = []
        rp_orbits = []
        rd_orbits = []
        for a in atoms:
            a = str(a)
            #if ith > 625:
            #    print a
            index = np.argwhere(a == orbitals.iloc[:,0])[0][0]
            rs = orbitals.iloc[index,1]
            rp = orbitals.iloc[index,2]
            rd = orbitals.iloc[index,3]
            rs_orbits.append(rs)
            rp_orbits.append(rp)
            rd_orbits.append(rd)
        rs_df.append(np.asarray(rs_orbits))
        rp_df.append(np.asarray(rp_orbits))
        rd_df.append(np.asarray(rd_orbits))
    orbit_df['rs'] = rs_df
    orbit_df['rp'] = rp_df
    orbit_df['rd'] = rd_df
    return orbit_df




def interaction(df):
    """ creates interaction terms from existing terms in dataframe"""
    for ith, i in enumerate(df):
        for jth, j in enumerate(df):
            if ith >= jth:
                i_label = i.replace(u'\u03b1','-')
                j_label = j.replace(u'\u03b1','-')
                i_label = i_label.replace(u'\u2013','-')
                j_label = j_label.replace(u'\u2013','-')
                i_label.encode('utf8')
                j_label.encode('utf8')
                label = '[' + str(i_label) + ']' + '_' + '[' + str(j_label) + ']'
                #print label
                df[label] = df[i].values*df[j].values
    return df
