# import functions
#
# Collects all improt functions

# import qmmlpack as qmml
# print(qmml.__doc__)
import matplotlib
from matplotlib import ticker

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
from pymatgen.ext.matproj import MPRester
m = MPRester("RK5GrTk1anSOmgAU")
import pymatgen as mg
import pickle
from alloy_functions import *

###
from matplotlib import ticker
from build_df import *

#import pymatgen as mp
from pymatgen.io.xyz import XYZ
from pymatgen import Lattice, Structure
from ase.io import read, write
#from pymatgen.io.vasp.inputs import Poscar
#from pymatgen.io.cifio import CifParser

import tqdm
import ast, itertools, hashlib

import os, math, sys
import IPython
import matplotlib as mplt
from tqdm import tqdm
