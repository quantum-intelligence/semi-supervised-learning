# magflips.py
# collection of codes to import magmom data, check for spin flips and import data
#

from import_functions import *
from spin_states import *
from alloy_functions import *
#import tensorflow as tf
import os
from sklearn.ensemble import ExtraTreesRegressor

from atomic_property import *
from mbtr import *
from fmcalc import *
from fmmlcalc_b import *
from get_vasp_data import *

from sklearn.model_selection import KFold
from pymatgen import Lattice, Structure, Molecule
from medeleev_functions import *
from ml_methods import *
from df_gen import *

from alloy_functions import *
from df_gen import *

import checkflip as cf

# load local_mag_mom files

#
# main codes :
#
class magflips:
    def __init__(self):
        self.dir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/energy_results/local_magmom_yiqi/'
        self.file = ['magmom_Te_fm_op.csv','magmom_Te_afm_op.csv','magmom_Se_fm_op.csv','magmom_Se_afm_op.csv',
                     'magmom_S_fm_op.csv','magmom_S_afm_op.csv']

    def localmu_gen(self):
        """ read files, get data """
        dir = self.dir
        file = self.file
        filestem = dir + file[0]
        df_localmu_Te_fm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_Te_fm = df_localmu_Te_fm.rename(columns={'mz_Cr':'mz_Cr_fm','mz_TM':'mz_TM_fm'})
        filestem = dir + file[1]
        df_localmu_Te_afm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_Te_afm = df_localmu_Te_afm.rename(columns={'mz_Cr':'mz_Cr_afm','mz_TM':'mz_TM_afm'})
        filestem = dir + file[2]
        df_localmu_Se_fm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_Se_fm = df_localmu_Se_fm.rename(columns={'mz_Cr':'mz_Cr_fm','mz_TM':'mz_TM_fm'})
        filestem = dir + file[3]
        df_localmu_Se_afm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_Se_afm = df_localmu_Se_afm.rename(columns={'mz_Cr':'mz_Cr_afm','mz_TM':'mz_TM_afm'})
        filestem = dir + file[4]
        df_localmu_S_fm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_S_fm = df_localmu_S_fm.rename(columns={'mz_Cr':'mz_Cr_fm','mz_TM':'mz_TM_fm'})
        filestem = dir + file[5]
        df_localmu_S_afm = pd.read_csv(filestem, usecols=[1,2,3,4,5,6,7])
        df_localmu_S_afm = df_localmu_S_afm.rename(columns={'mz_Cr':'mz_Cr_afm','mz_TM':'mz_TM_afm'})
        # check spin flips using python scripts
        # from fm to afm
        df_localmu_Te_fm = cf.fm_afm_flip(df_localmu_Te_fm)
        df_localmu_Se_fm = cf.fm_afm_flip(df_localmu_Se_fm)
        df_localmu_S_fm = cf.fm_afm_flip(df_localmu_S_fm)
        # from afm to fm
        df_localmu_Te_afm = cf.afm_fm_flip(df_localmu_Te_afm)
        df_localmu_Se_afm = cf.afm_fm_flip(df_localmu_Se_afm)
        df_localmu_S_afm = cf.afm_fm_flip(df_localmu_S_afm)
        fm_afm_flipped_Te = df_localmu_Te_fm['formula'][df_localmu_Te_fm['fm_afm_flipped'] == -1].values
        fm_afm_flipped_Se = df_localmu_Se_fm['formula'][df_localmu_Se_fm['fm_afm_flipped'] == -1].values
        fm_afm_flipped_S = df_localmu_S_fm['formula'][df_localmu_S_fm['fm_afm_flipped'] == -1].values
        fm_afm_flipped = list(fm_afm_flipped_Te) + list(fm_afm_flipped_Se) + list(fm_afm_flipped_S)
        afm_fm_flip_Te = df_localmu_Te_afm['formula'][df_localmu_Te_afm['afm_fm_flipped'] == -1].values
        afm_fm_flip_Se = df_localmu_Se_afm['formula'][df_localmu_Se_afm['afm_fm_flipped'] == -1].values
        afm_fm_flip_S = df_localmu_S_afm['formula'][df_localmu_S_afm['afm_fm_flipped'] == -1].values
        afm_fm_flip =  list(afm_fm_flip_Te) + list(afm_fm_flip_Se) + list(afm_fm_flip_S)
        fm_afm_x = [fm_afm_flipped_Te, fm_afm_flipped_Se, fm_afm_flipped_S]
        afm_fm_x = [afm_fm_flip_Te, afm_fm_flip_Se, afm_fm_flip_S]
        return fm_afm_flipped, afm_fm_flip, fm_afm_x, afm_fm_x


    def get_double_flip(self, fm_afm_flipped, afm_fm_flip):
        """
            function to get afm_fm and fm_afm or 'double flips'
        """
        double_flip = []
        for i in fm_afm_flipped:
            if i in afm_fm_flip:
                double_flip.append(i)
                print(i)
        # print('----')
        # for i in afm_fm_flip:
        #     if i in fm_afm_flipped:
        #         print(i)
        for s in double_flip:
            afm_fm_flip.remove(s)
            fm_afm_flipped.remove(s)
        return double_flip, fm_afm_flipped, afm_fm_flip



    def check_remove(self, list_X, double_flip):
        """
           # check and remove from list
        """
        list_X = list(list_X)
        for s in double_flip:
            if s in list_X:
                list_X.remove(s)
        list_X = np.asarray(list_X)
        return list_X


# double_flip, fm_afm_flipped, afm_fm_flip = get_double_flip(fm_afm_flipped, afm_fm_flip)
#
# # Remove double_flip from all the lists after checking carefully
#
# fm_afm_flipped_Te = check_remove(fm_afm_flipped_Te, double_flip)
# fm_afm_flipped_Se = check_remove(fm_afm_flipped_Se, double_flip)
# fm_afm_flipped_S = check_remove(fm_afm_flipped_S, double_flip)
#
# afm_fm_flip_Te = check_remove(afm_fm_flip_Te, double_flip)
# afm_fm_flip_Se = check_remove(afm_fm_flip_Se, double_flip)
# afm_fm_flip_S = check_remove(afm_fm_flip_S, double_flip)
