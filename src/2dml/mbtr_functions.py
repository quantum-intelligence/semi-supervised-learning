# Functions for MBTR_catalysis.ipynb
import re
import pandas as pd
import numpy as np
#import pubchempy as pcp
from sklearn import preprocessing



def get_bond_info(bond):
    """ 
        convert bond info from excel file input
    """
    elements = re.findall(r'\b[A-Z][a-z]?\b',bond)
    bond_order = re.findall('[\W]',bond)[0]
    #print bond_order
    #'\u2261'
    bond_info = []
    bond_order_num = 0
    if bond_order == '-':
        bond_order_num = 1
    elif bond_order == '=':
        bond_order_num = 2
    else: #must be triple bond
        bond_order_num = 3
    #if bond_order == '\u2261':
    #    bond_order_num = 3
    #if bond_order == 
    #    bond_order_num = 3
    bond_info = elements
    bond_info.append(bond_order_num)
    #print 'je0, ',(elements)
    #print bond_order_num
    #print bond_info
    return bond_info

def convert_bondinfo(bondenergy):
    """Created new df with updated bond information"""
    bond_list = []
    for bond in bondenergy['Bond']:
        #bond_info = [get_bond_info(bond)]
        bond_info = unicode(get_bond_info(bond))
        bond_list.append(bond_info)
    bond_df = pd.DataFrame(bond_list,index=bondenergy.index,columns=['bond_info'])
    new_df = pd.concat([bondenergy, bond_df],axis=1 )
    return new_df


def get_bondenergy(bond_df,pcp_bonds):
    """
        input : pcp_bonds
        output : bond energies
        Use bond info converted from pcp bond object and look up table
        from xcel file with bond energies to create feature vector
        with list of bond energies correspondign to input molecule
    """
    e_vals = []
    for bond in pcp_bonds:
        # ind = np.argwhere(b == dd['bond_info'])
        # print (bond), type(bond), len(bond)
        uni_bond = unicode(bond)
        dex = bond_df.loc[bond_df['bond_info'] == uni_bond].index
        if len(dex) == 0:
            bond_rev = bond
            bond0 = bond[0]; bond1 = bond[1]
            bond_rev[0] = bond1
            bond_rev[1] = bond0
            bond_rev = unicode(bond_rev)
            dex = bond_df.loc[bond_df['bond_info'] == bond_rev].index
            val = bond_df.loc[dex,['D(kJ/mol)']].values[0][0]
            # print val
            e_vals.append(val)
        else:
            val = bond_df.loc[dex,['D(kJ/mol)']].values[0][0]
            # print val
            e_vals.append(val) 
    return e_vals


def get_molcoords(molecule):
    """  
        Inputs a pcp molecule and converts this to a list of coordinates
        and atoms for use with creating a pymatgen object
    """
    me_atoms = molecule.atoms
    molec = me_atoms
    coords = []
    atom_bag = []
    Z_list = []
    for atom in molec:
        #print m.element
        atom_bag.append(atom.element)
        Z_list.append(atom.number)
        xx = atom.x
        yy = atom.y
        zz = 0.0
        coords.append([xx,yy,zz])
    return atom_bag, coords, Z_list


def get_info_cid(df,colname,pubfp):
    """ 
        Gets info using pubchempy CID number.
        e.g colname = 'Reactant 1 CID'
        Outputs iupac name instead of molecular formula
    """ 
    if pubfp == True:
        dummy = [u'00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000']
    else:
        dummy =('00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')
    formulas = []
    weights = []
    complexity = []
    fps = []
    for i, row in enumerate(df.iloc[:,0]):
        #print i
        chemname = df[colname][i]
        #print chemname, type(chemname)
        if np.isnan(chemname):
            #print 'oops!!'
            #print chemname
            formulas.append(np.nan)
            #
            #fill absence of molecule with zero values not nan
            weights.append(0)
            fps.append(dummy) #length of fingerprint
            #fps.append(np.zeros(230))
            complexity.append(0)
        else:
            chemname = int(chemname)
            #print 'gpt here', chemname
            c_cid =  pcp.get_compounds(chemname,'cid')
            #print c_cid
            formula = c_cid[0].iupac_name
            #formula = c_cid[0].molecular_formula
            weight = c_cid[0].molecular_weight
            if pubfp == True:
                fp = c_cid[0].cactvs_fingerprint
            else:
                fp = c_cid[0].fingerprint
            rcomplex = c_cid[0].complexity
            formulas.append(formula)
            weights.append(weight)
            fps.append(fp)
            complexity.append(rcomplex)
            #c = pcp.Compound.from_cid(c_cid)
    return formulas, weights, fps, complexity



#get_subprints(df2,'Reactant 1')


def get_pcp_bonds(pcp_molec):
    mol_atoms = pcp_molec.atoms
    #print 'doc', (mol_atoms[0].to_dict())
    mol_bonds = pcp_molec.bonds
    #print 'atoms', mol_atoms
    mol_bonds
    pcp_bond_info = []
    for mol_bond in mol_bonds:
        #print mol_bond
        aid1 = mol_bond.aid1
        aid2 = mol_bond.aid2
        #print aid1 
        for mol_atom in mol_atoms:
            dict_atom = mol_atom.to_dict()
            if aid1 == dict_atom['aid']:
                elem_aid1 = dict_atom.get('element')
            if aid2 == dict_atom['aid']:
                elem_aid2 = dict_atom.get('element')
        bond_list = [elem_aid1,elem_aid2,mol_bond.order]
        #bond_list = unicode(bond_list)
        #Dont convert to unicoe here.. need to be able to swap positions easily
        pcp_bond_info.append(bond_list)
    return pcp_bond_info


    
def cid_to_name(df2, species_cid): 
    """
        write function to get IUPAC names from CID numbers
    """
    names = []
    for ith in np.arange(len(df2)):
        ith_cid = df2[species_cid][ith]
        #print ith_cid, type(ith_cid)
        if np.isnan(ith_cid):
            names.append(np.nan)
        else:
            ith_cid = int(ith_cid)
            cmpd = pcp.get_compounds(ith_cid,namespace=u'cid')[0]
            name = cmpd.iupac_name
            #print ith, ith_cid, name
            names.append(name)
    return names

