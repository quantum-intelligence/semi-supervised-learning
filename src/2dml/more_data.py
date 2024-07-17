# get more data for energy and magnetic moment energy_results_spin_so_S
from import_functions import *


def master_energy_df(main_dir, edir,recalculate):
    """
       gathers all energy data for new dataset
       returns:
           edf, [df_spin_so_Te, df_spin_so_Se, df_spin_so_S,
                 df_afm_so_Te, df_afm_so_Se, df_afm_so_S]
    """
    # recalculate = False
    efile_spin_so_Te = edir + '/energy_results_spin_so_Te.csv'
    efile_afm_so_Te = edir + '/energy_results_afm_Te.csv'
    df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
    df_afm_so_Te = pd.read_csv(efile_afm_so_Te, delimiter=',', usecols=[1,2])
    # Calculate cohesive energies - Te
    df_elements = get_unique_elem_info(main_dir, df_spin_so_Te, recalculate)
    df_spin_so_Te = gen_cohesive(df_spin_so_Te, df_elements)
    df_afm_so_Te = gen_cohesive(df_afm_so_Te, df_elements)

    # efile_afm_so_Se = edir + '/energy_results_afm_Se.csv'
    # efile_spin_so_Se = edir + '/energy_results_spin_so_Se.csv'
    # df_afm_so_Se = pd.read_csv(efile_afm_so_Se, delimiter=',', usecols=[1,2])
    # df_spin_so_Se = pd.read_csv(efile_spin_so_Se, delimiter=',', usecols=[1,2])
    # # Calculate cohesive energies - Se
    # df_elements = get_unique_elem_info(df_nospin_Se, recalculate=recalculate)
    # df_spin_so_Se = gen_cohesive(df_spin_so_Se, df_elements)
    # df_afm_so_Se = gen_cohesive(df_afm_so_Se, df_elements)

    # efile_afm_so_S = edir + '/energy_results_afm_S.csv'
    # efile_spin_so_S = edir + '/energy_results_spin_so_S.csv'
    # df_afm_so_S = pd.read_csv(efile_afm_so_S, delimiter=',', usecols=[1,2])
    # df_spin_so_S = pd.read_csv(efile_spin_so_S, delimiter=',', usecols=[1,2])
    # # Calculate cohesive energies - S
    # df_elements = get_unique_elem_info(df_spin_S, recalculate=recalculate)
    # #df_nospin_Se = gen_cohesive(df_nospin_Se, df_elements)
    # df_spin_so_S = gen_cohesive(df_spin_so_S, df_elements)
    # df_afm_so_S = gen_cohesive(df_afm_so_S, df_elements)

    #CONCATENATE different spin ocnfigurations & relabel columns
    # edf_spin_so = pd.concat((df_spin_so_Te, df_spin_so_Se, df_spin_so_S))
    edf_spin_so = df_spin_so_Te.copy()
    edf_spin_so = edf_spin_so.rename(columns={'energy':'energy_spin_so'})
    edf_spin_so = edf_spin_so.rename(columns={'cohesive':'cohesive_spin_so'})
    edf_spin_so = edf_spin_so.rename(columns={'formula':'formula_spin_so'})
    edf_spin_so = edf_spin_so.sort_values('formula_spin_so')

    # edf_afm_so = pd.concat((df_afm_so_Te, df_afm_so_Se, df_afm_so_S))
    edf_afm_so = df_afm_so_Te.copy()
    edf_afm_so = edf_afm_so.rename(columns={'energy':'energy_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'cohesive':'cohesive_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'formula':'formula_afm'})
    edf_afm_so = edf_afm_so.rename(columns={'total_elem_energy':'total_elem_energy_afm'})
    edf_afm_so = edf_afm_so.sort_values('formula_afm')

    edf = pd.concat((edf_spin_so, edf_afm_so),axis=1)
    edf = edf.drop(columns=['formula_spin_so'])#,'total_elem_energy'])
    edf = edf.rename(columns={'formula_afm':'formula_edfm'})
    edf = edf.rename(columns={'elem_list_spin_so':'elem_list'})
    edf = edf.reset_index() #sort_values alters index which causes problem later if use concat()
    return edf, [df_spin_so_Te,df_afm_so_Te], df_elements




def master_mag_df(edir):
    """ gather magnetic moment info for all spin configurations"""
    #edir = '/Users/trevorrhone/Documents/Kaxiras/2DML/Alloys_ML/energy_results'

    # efile_spin_so_S = edir + '/magmom_results_spin_so_S.csv'
    # df_spin_so_S = pd.read_csv (efile_spin_so_S, delimiter=',', usecols=[1,2])
    # df_spin_so_S = df_spin_so_S.rename(columns={'mag_mom':'magmom_spin_so'})
    # df_spin_so_S = df_spin_so_S.rename(columns={'formula':'formula_spin_so'})
    # #
    # efile_spin_so_Se = edir + '/magmom_results_spin_so_Se.csv'
    # df_spin_so_Se = pd.read_csv (efile_spin_so_Se, delimiter=',', usecols=[1,2])
    # df_spin_so_Se = df_spin_so_Se.rename(columns={'mag_mom':'magmom_spin_so'})
    # df_spin_so_Se = df_spin_so_Se.rename(columns={'formula':'formula_spin_so'})
    #
    efile_spin_so_Te = edir + '/magmom_results_spin_so_Te.csv'
    df_spin_so_Te = pd.read_csv(efile_spin_so_Te, delimiter=',', usecols=[1,2])
    df_spin_so_Te = df_spin_so_Te.rename(columns={'mag_mom':'magmom_spin_so'})
    df_spin_so_Te = df_spin_so_Te.rename(columns={'formula':'formula_spin_so'})
    df_spin_so_Te = df_spin_so_Te.sort_values('formula_spin_so')

    efile_afm_Te = edir + '/magmom_results_afm_Te.csv'
    df_afm_Te = pd.read_csv (efile_afm_Te, delimiter=',', usecols=[1,2])
    df_afm_Te = df_afm_Te.rename(columns={'mag_mom':'magmom_afm'})
    df_afm_Te = df_afm_Te.rename(columns={'formula':'formula_afm'})
    df_afm_Te = df_afm_Te.sort_values('formula_afm')
    #
    # efile_afm_Se = edir + '/magmom_results_afm_Se.csv'
    # df_afm_Se = pd.read_csv (efile_afm_Se, delimiter=',', usecols=[1,2])
    # df_afm_Se = df_afm_Se.rename(columns={'mag_mom':'magmom_afm'})
    # df_afm_Se = df_afm_Se.rename(columns={'formula':'formula_afm'})
    # #
    # efile_afm_S = edir + '/magmom_results_afm_S.csv'
    # df_afm_S = pd.read_csv (efile_afm_S, delimiter=',', usecols=[1,2])
    # df_afm_S = df_afm_S.rename(columns={'mag_mom':'magmom_afm'})
    # df_afm_S = df_afm_S.rename(columns={'formula':'formula_afm'})

    # df_spin_so = pd.concat((df_spin_so_Te,df_spin_so_Se,df_spin_so_S))
    # df_afm = pd.concat((df_afm_Te, df_afm_Se, df_afm_S))

    df_spin_so = df_spin_so_Te.copy()
    df_afm = df_afm_Te.copy()

    df_mag_tot = pd.concat((df_spin_so, df_afm), axis=1)
    formula = df_spin_so['formula_spin_so'].values

    df_mag_tot['formula'] = formula
    df_mag_tot = df_mag_tot.reset_index()
    df_mag_tot = df_mag_tot.drop(columns=['index'])
    return df_mag_tot
