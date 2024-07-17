#
# checkflip.py
#

import numpy as np
import pandas as pd


def afm_fm_flip(df_afm_input):
    """
        check afm_to_fm spin flip
        - provided know calculation was initialized with antiparallel spins
        - check to see if spins are parallel in the end which means that they flipped.
    """
    # df_afm_Se.columns
    df_afm = df_afm_input.copy()
    mag_Cr_arr = []
    mag_TM_arr = []
    flip_arr = []
    for row in df_afm.iterrows():
        mag_Cr = row[1]['mz_Cr_afm']
        mag_TM = row[1]['mz_TM_afm']
        spin_flip = mag_Cr*mag_TM
        if spin_flip > 0:
            flipped = -1
        else:
            flipped = 1
        # print(flipped, mag_Cr, mag_TM)
        flip_arr.append(flipped)
        # mag_Cr_arr.append(mag_Cr)
        # mag_TM_arr.append(mag_TM)
    flip_arr  = np.asarray(flip_arr)
    df_afm['afm_fm_flipped'] = flip_arr
    return df_afm



def fm_afm_flip(df_fm_input):
    """
        check fm_to_afm spin flip
        - provided know calculation was initialized with antiparallel spins
        - check to see if spins are parallel in the end which means that they flipped.
    """
    df_fm = df_fm_input.copy()
    mag_Cr_arr = []
    mag_TM_arr = []
    flip_arr = []
    for row in df_fm.iterrows():
        mag_Cr = row[1]['mz_Cr_fm']
        mag_TM = row[1]['mz_TM_fm']
        spin_flip = mag_Cr*mag_TM
        if spin_flip < 0:
            flipped = -1
        else:
            flipped = 1
        flip_arr.append(flipped)
    flip_arr  = np.asarray(flip_arr)
    df_fm['fm_afm_flipped'] = flip_arr
    return df_fm
