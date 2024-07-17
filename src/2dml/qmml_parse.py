import os
import shutil
import pymatgen as mp
from pymatgen.io.xyz import XYZ
from pymatgen import Lattice, Structure
from ase.io import read, write
import qmmlpack as qmml

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

import pymatgen as mg
from mendeleev import element
from fractions import Fraction



def create_xyz_input(mydir):
    """
        creates xyz input, uses ase, can have periodicity saved to extneded xyz format
        - set read directory
    """

    #mydir = '/Users/trevorrhone/Documents/Kaxiras/mbtr/test_poscar/contcar_folder'
    poscars = os.listdir(mydir)
    #print(poscars)
    cmpd_labels = [x.split('_')[0] for x in poscars]
    #print(cmpd_labels)
    dext = '/Users/trevorrhone/Documents/Kaxiras/mbtr/xyz_dir'
    main_dir = '/Users/trevorrhone/Documents/Kaxiras/mbtr/'
    os.chdir(main_dir)
    for poscar in poscars:
        file = (os.path.join(mydir,poscar))
        #print('input file to ase poscar',file)
        ase_poscar = read(file)
        #print(ase_poscar)
        #ABX_monolayer = Structure.from_file(file)
        #xyz_data = XYZRhone(ABX_monolayer)
        dfile = os.path.join(dext, poscar)+'.xyz'
        #print('saving xyz: ', dfile)
        #print(type(dfile),'\n',dfile)
        os.chdir(dext)
        #print('PRINTING', poscar, type(poscar))
        writefile=str(poscar)
        #ase_poscar.write('MARIOS.xyz',format='extxyz')#,format='extxyz')
        ase_poscar.write(writefile+'.xyz',format='extxyz')#,format='extxyz')
        #xyz_data.write_file(dfile)

    files = os.listdir('/Users/trevorrhone/Documents/Kaxiras/mbtr/xyz_dir')
    files = [x for x in files if '.DS' not in x]
    files = [x for x in files if 'result' not in x]
    file_ext = '/Users/trevorrhone/Documents/Kaxiras/mbtr/xyz_dir/'
    #print(files)
    #print('testing', type('testing'))
    with open( 'result.xyz', 'w' ) as result:
        for file_ in files:
            #print('FILE IS: ', file_)
            file_ = file_ext + file_
            with open( file_, 'r') as f:
                lines = f.readlines()
                last = lines[-1]
                for line in lines[:-1]:
                    result.write( line )
                result.write(last)
                result.write('\n') # can't 'handle two spaces \n

    # concat = ''.join([open(f).read() for f in files])
    # print(base_dir)
    # print(file_ext)
    testfilename = os.path.join(dext, 'result.xyz');
    # testfilename = os.path.join(base_dir, 'mytestxyz.xyz');

    print('test file name: ', testfilename)
    #testfilename = os.path.join(file_ext, 'result.xyz');
    cgtraw =  qmml.import_extxyz(testfilename, additional_properties=True);
    os.chdir(main_dir)
    return cgtraw, cmpd_labels


def get_abc(abc,i):
    """ 
        parse lattice parameters from xyz file
    """
    a = []
    b = []
    c = []
    #print(abc[0])
    for item in abc[i][:3][:]:
        #print(item)
        if type(item) == str:
            vals = item.split('"')
            #print(vals)
            x = vals[1]
            a.append(np.float(x))
        else:
            a.append(item)
    for item in abc[i][3:6][:]:
        #print(item)
        if type(item) == str:
            vals = item.split('"')
            #print(vals)
            x = vals[1]
            b.append(np.float(x))
        else:
            b.append(item)
    #print(' ---------- ')
    for item in abc[i][6:9][:]:
        #print(item)
        if type(item) == str:
            #print('gets here', item)
            vals = item.replace('"','')
            #print(vals)
            c.append(np.float(vals))
        else:
            c.append(item)
    lattice_vec = [a, b, c]
    return lattice_vec


def lat_vec_gen(cgtraw):
    """ parse lattice vector information"""
    abc = [s.mp[0:9] for s in cgtraw]
    # i = 0
    lattice_vectors = []
    for i in np.arange(len(abc)):
        lattice_vec = get_abc(abc,i)
        lattice_vectors.append(lattice_vec)
    return lattice_vectors


def dataset_info(z, r, e, basis=None, verbose=None):
    """Information about a dataset.
    
    Returns a dictionary containing information about a dataset.
    
    Parameters:
      z - atomic numbers
      r - atom coordinates, in Angstrom
      e - energies
      basis - basis vectors for periodic systems
      verbose - if True, also prints the information
    
    Information:
      elements - elements occurring in dataset
      max_elements_per_system - largest number of different elements in a system
      max_same_element_per_system - largest number of same-element atoms in a system
      max_atoms_per_system - largest number of atoms in a system
      min_distance - minimum distance between atoms in a system
      max_distance - maximum distance between atoms in a system    
    """
    assert len(z) == len(r) == len(e)
    assert basis is None or len(basis) == len(z)

    i = {}
    
    i['number_systems'] = len(z)
    
    # elements
    i['elements'] = np.unique(np.asarray([a for s in z for a in s], dtype=np.int)) 
    i['max_elements_per_system'] = max([np.nonzero(np.bincount(s))[0].size for s in z])
    i['max_same_element_per_system'] = max([max(np.bincount(s)) for s in z]) 
    
    # systems
    i['max_atoms_per_system'] = max([len(s) for s in z])
    i['systems_per_element'] = np.asarray([np.sum([1 for m in z if el in m]) for el in range(118)], dtype=np.int)
    
    # distances
    assert len(r) > 0
    dists = [qmml.lower_triangular_part(qmml.distance_euclidean(rr), -1) for rr in r]
    i['min_distance'] = min([min(d) for d in dists if len(d) > 0])
    i['max_distance'] = max([max(d) for d in dists if len(d) > 0])
    
    # verbose output
    if verbose:
        if basis is None: print('{} finite systems (molecules)'.format(i['number_systems']))
        else: print( '{} periodic systems (materials)'.format(i['number_systems']) )    
        print('elements: {} ({})'.format(' '.join([qmml.element_data(el, 'abbreviation') \
            for el in i['elements']]), len(i['elements'])))
        print('max #els/system: {};  max #el/system: {};  max #atoms/system: {}'.format( \
            i['max_elements_per_system'], i['max_same_element_per_system'], i['max_atoms_per_system']))
        print('min dist: {:3.2f};  max dist: {:3.2f};  1/min dist: {:3.2f};  1/max dist: {:3.2f}'.format( \
            i['min_distance'], i['max_distance'], 1./i['min_distance'], 1./i['max_distance']))
        
    return i




def get_contcar(targetdir,spinlabel):
    """ collect CONTCAR files on stampede to a folder """
    curr_dir = os.getcwd()   
    store_dir = os.path.join(curr_dir,'storage')
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    print(curr_dir)
    # targetdir = 'monolayer3'
    targetdir = curr_dir + '/' + targetdir
    dirs = os.listdir(targetdir)
    dirs = [x for x in dirs if '.' not in x]

    for dir in dirs:
        # dir = os.path.join(targetdir,dir)
        src = targetdir + '/' + dir + '/' + spinlabel + '/CONTCAR'
        if not os.path.exists(src):
            print('calculatino incomplete for ', dir)
        else:
            dst_dir = os.path.join(store_dir, dir + '/' + spinlabel)
            dst = os.path.join(store_dir, dir + '/' + spinlabel + '/CONTCAR')
            #print('DST',dst,'\n')
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            else:
                print('exists already', dst_dir)
            print('source: ', src)
            print('dst : ', dst, '\n')
            shutil.copy(src,dst)