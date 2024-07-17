# Property_vector for magnetic moment prediction and DeltaE(FM,AFM) prediction
from import_functions import *

class property_vector:

    def __init__(self, df):
        #df = pickle.load( open( 'df.p', "rb" ) )
        self.df = df

    def property_vector_abx_gen(self, species):
        """
            parses list of mendeleev elements to vectors of their
            corresponding atomic properties
        """
        atomic_number_list = []
        atomic_radius_list = []
        atomic_radius_rahm_list = []
        covalent_radius_pyykko_list = []
        dipole_polarizability_list = []
        electron_affinity_list = []
        en_allen_list = []
        gas_basicity_list = []
        crystal_radius_list = []
        ionic_radius_list = []
        nvalence_list = []
        hardness_list = []
        #
        boiling_point_list = []
        density_list = []
        en_list = []
        electrons_list = []
        ione_list = []
        ionic_rad_list = []
        proton_affinity_list = []
        softness_list = []
        zeff_list =[]
        #
        for elem in species:
            element(str(elem))
            mdlv = element(str(elem))
            m_atomic_number = mdlv.atomic_number
            m_atomic_radius = mdlv.atomic_radius
            m_atomic_radius_rahm = mdlv.atomic_radius_rahm
            m_covalent_radius_pyykko = mdlv.covalent_radius_pyykko
            m_dipole_polarizability = mdlv.dipole_polarizability
            m_electron_affinity = mdlv.electron_affinity
            m_en_allen = mdlv.en_allen
            m_gas_basicity = mdlv.gas_basicity
            m_ionic_radius, m_crystal_radius = get_ionic_crystal_r(mdlv)
            m_nvalence = mdlv.nvalence()
            m_hardness = mdlv.hardness()
            # m_crystal_radius = mdlv.crystal_radius
            # m_ionic_radius = mdlv.ionic_radius
            boiling_point = mdlv.boiling_point
            density = mdlv.density
            electronegativity = mdlv.electronegativity()
            electrons = mdlv.electrons
            ionenergies = mdlv.ionenergies[1]
            ionic_radius = mdlv.ionic_radii[0].ionic_radius
            proton_affinity = mdlv.proton_affinity
            softness = mdlv.softness()
            zeff = mdlv.zeff()
            #
            boiling_point_list.append(boiling_point)
            density_list.append(density)
            en_list.append(electronegativity)
            electrons_list.append(electrons)
            ione_list.append(ionenergies)
            ionic_rad_list.append(ionic_radius)
            proton_affinity_list.append(proton_affinity)
            softness_list.append(softness)
            zeff_list.append(zeff)
            #
            atomic_number_list.append( m_atomic_number )
            atomic_radius_list.append( m_atomic_radius )
            atomic_radius_rahm_list.append( m_atomic_radius_rahm )
            covalent_radius_pyykko_list.append( m_covalent_radius_pyykko )
            dipole_polarizability_list.append( m_dipole_polarizability )
            electron_affinity_list.append( m_electron_affinity )
            en_allen_list.append( m_en_allen )
            gas_basicity_list.append( m_gas_basicity )
            crystal_radius_list.append( m_gas_basicity )
            ionic_radius_list.append( m_ionic_radius )
            nvalence_list.append(m_nvalence)
            hardness_list.append(m_hardness)
        # ORIGINAL results_list:
        # results_list = [atomic_number_list, atomic_radius_list, atomic_radius_rahm_list,
        #                     covalent_radius_pyykko_list, dipole_polarizability_list, electron_affinity_list,
        #                     en_allen_list, gas_basicity_list, crystal_radius_list,
        #                     ionic_radius_list, nvalence_list, hardness_list ,
        #                     boiling_point_list, density_list, en_list,
        #                     electrons_list, ione_list, ionic_rad_list,
        #                     proton_affinity_list, softness_list, zeff_list ]
        # SHORTENED RESULTS_LIST:
        results_list = [atomic_number_list, atomic_radius_list,
                            covalent_radius_pyykko_list, dipole_polarizability_list, electron_affinity_list,
                            en_allen_list, gas_basicity_list,
                            ionic_radius_list, nvalence_list, hardness_list,
                            density_list, en_list,
                            electrons_list, ione_list, ionic_rad_list,
                            softness_list ]
        results_vector = []
        for results in results_list:
            for entry in results:
                # remove 'NONE' from property vettor
                if entry == None:
                    results_vector.append(0)
                else:
                    results_vector.append(entry)
        return results_vector





    def abx_vector_gen(self, elem_list):
        """ create 5 element abx vector using df['elem_list'] entry """
        abx_vector = np.empty(5,dtype=object)
        cr = mg.Element('Cr')
        abx_vector[:2] = cr
        b_counter = 0
        b_sites = np.empty(2,dtype=object)
        b_z_vector = np.empty(2,dtype=object)
        for elem in elem_list:
            is_Cr = elem == cr
            if elem.is_transition_metal and not is_Cr:
                abx_vector[1] = elem
            elif elem.is_chalcogen:
                abx_vector[4] = elem
            elif elem == cr:
                abx_vector[1] = cr
            else:
                if b_counter == 0:
                    b_sites[:] = elem
                    b_z_vector[:] = elem.number
                    b_counter = b_counter + 1
                    #print('b--sites', b_sites)
                elif b_counter == 1:
                    b_sites[1] = elem
                    b_z_vector[1] = elem.number
        print('elem_list',elem_list)
        print('b_z_vector',b_z_vector)
        b_order = np.argsort(b_z_vector)
        b_sites = b_sites[b_order]
        abx_vector[2:4] = b_sites
        return abx_vector


    def get_abx_vector_list(self, df):
        """
            calculate list of sum of energies of constituent atoms of compounds
        """
        abx_vector_list = []
        for elem_list in df['elem_list']:
            abx_vector = abx_vector_gen(elem_list)
            abx_vector_list.append(abx_vector)
        return abx_vector_list



    def gen_property_vector_list(self, df):
        """generates list of property fectors"""
        pvec_list = []
        for elem_list in df['elem_list']:
            #convert to five element vector
            elem_list = self.abx_vector_gen(elem_list)
            pvec = self.property_vector_abx_gen(elem_list)
            #print('len(pvec)', len(pvec))
            pvec = np.asarray(pvec)
            pvec_list.append(pvec)
        pvec_list = np.asarray(pvec_list)
        return pvec_list

    # NOTES: abx_vector_gen(elem_list) and
    #        abx_vector_list = get_abx_vector_list(df)
    #        to generate vector of elements in appropriate format
    # Creata  a vector of properties (p1: A1 B1 X, p2: A B X, p3: A B X, etc)
    #
    # Create a list of propoerty_vectors according to df
    #
