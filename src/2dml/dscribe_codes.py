#
# Code for generating descriptors written here'
#


def find_unique_elems(data):
    """
        find unique elements using pymatgent
        - fast implementation
    """
    all_elems = []
    for f in data['formula']:
        comp = mp.Composition(f)
        elems = comp.elements
        elems = [str(x) for x in elems]
        all_elems.extend(elems)
    #print(all_elems)
    unique_elems = np.unique(all_elems)
    return unique_elems



def soap_featurize(ase_atom, unique_elems):
    """
        create soap descriptor
        # Periodic systems
    """
    rcut = 1.1
    nmax = 2
    lmax = 1
    unique_elems = list(unique_elems)
    periodic_soap = SOAP(
        species=unique_elems, #[29],
        rcut=rcut,
        nmax=nmax,
        lmax=nmax,
        periodic=True,
        average=True,
        sparse=False
    )

    if not pd.isnull(ase_atom):
        soap_desc = periodic_soap.create(ase_atom)
    else:
        print('have nan ase_atom')
        return np.nan
    return soap_desc




def create_ref_tmx():
    """
        # initalize ase object wtih reference POSCAR
    """
    ref_cri3 = ase.io.read(filename='vasp/POSCAR')
    # ref_cri3.get_chemical_formula()
    ref_positions = ref_cri3.get_positions()
    ref_symbols = ref_cri3.symbols
    ref_cell = ref_cri3.cell
    return ref_cell


def gen_ase_tmx(ref_cell,energy_df):
    """ generate list of ase structures using df['formula'] and ref ase from POSCAR """
    tmx_atom_set = []
    for tmx_formula in energy_df['formula'][:]:
        if tmx_formula == 'Ru4Cl12K10':
            print('error in formula', 'Ru4Cl12K10')
        else:
            #print(tmx_formula)
            # tmx_formula = "Cr2Ru2I12"
            # ref_atoms.set_chemical_symbols(tmx_formula)
            tmx_atoms = Atoms(tmx_formula, ref_positions)
            tmx_atoms.set_cell(ref_cell)
            tmx_atom_set.append(tmx_atoms)
            # dir(atoms)
    return tmx_atom_set

def gen_tmx_soap(tmx_atom_set, unique_elems):
    """ Create list of soap descriptors: """
    tmx_soap_list = []
    for i in np.arange(len(tmx_atom_set[:])):
        # print(i, tmx_atom_i)
        tmx_atom_i = tmx_atom_set[i]
        tmx_soap = soap_featurize(tmx_atom_i, unique_elems)
        # cri3_soap = soap_featurize(cri3, unique_elems)
        tmx_soap_list.append(tmx_soap)
    return tmx_soap_list
