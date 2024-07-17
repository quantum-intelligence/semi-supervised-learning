# Scripts for extracting data from Theta calculations

from import_functions import *


def relaxed_energy(data):
    """ extract final total energy """
    wf = ['initial','spin','afm','spin_so','afm_so']
    formula = data.keys()
    formula = list(formula)
    initial = []
    spin = []
    afm = []
    spin_so = []
    afm_so = []
    wf_dict =  {'initial':initial, 'spin':spin, 'afm':afm, 'spin_so':spin_so, 'afm_so':afm_so}
    energy_df = pd.DataFrame()
    energy_df['formula'] = formula
    for w in wf:
        #print(w)
        w_list = []
        for f in formula:
            #print(w, f)
            #wf_dict[w].
            subkeys = [k for k in data[f].keys()]
            if w in subkeys:
                eval = data[f][w]['energy']
                if type(eval) == list:
                    #print(eval, eval[-1])
                    eval = eval[-1]
                #else:
                #    print(eval)
            else:
                eval = np.nan
            w_list.append(eval)
        #print(w)
        energy_df[w] = w_list
    return energy_df



def pull_local_mag(data, abx, wf, so):
    """
       extract magnetization data
    - if not so calc, just get last x value in listed
    - if is spin orbit calc, then need to get x, y, z info
    """
    datatag = 'magnetization'
    if so == False:
        magdat = data[abx][wf][datatag][-1]
        orb = magdat[1]
        mag_i = magdat[0]
        maginfo = magdat[2]
        magsites = np.arange(len(maginfo))
        mag_tot = [x[-1] for x in maginfo]
        magsites_tot = {}
        for i in (magsites):
            magsites_tot[i] = mag_tot[i]
        return [magsites_tot]
    else:
        magdat_x = data[abx][wf][datatag][-3][2]
        magdat_y = data[abx][wf][datatag][-2][2]
        magdat_z = data[abx][wf][datatag][-1][2]
        magsites = np.arange(len(magdat_x))
        magx_tot = [x[-1] for x in magdat_x]
        magy_tot = [x[-1] for x in magdat_y]
        magz_tot = [x[-1] for x in magdat_z]
        magsitesx_tot = {}
        magsitesy_tot = {}
        magsitesz_tot = {}
        for i in (magsites):
            magsitesx_tot[i] = magx_tot[i]
            magsitesy_tot[i] = magy_tot[i]
            magsitesz_tot[i] = magz_tot[i]
        return [magsitesx_tot, magsitesy_tot, magsitesz_tot]



def gen_local_mag(data):
    """
        extract local mag mom from JSON output
    """
    formula = data.keys()
    formula = list(formula)
    wflist = ['initial','spin','afm','spin_so','afm_so']
    initial = []
    spin = []
    afm = []
    spin_so = []
    afm_so = []
    mag_df = pd.DataFrame()
    mag_df['formula_mag'] = formula
    wf_dict =  {'initial':initial,'spin':spin,'afm':afm,'spin_so':spin_so,'afm_so':afm_so}
    for wf in wflist:
        w_list = []
        for abx in formula:
            subkeys = [k for k in data[abx].keys()]
            if wf in subkeys:
                #print('wf', wf)
                so = 'so' in wf
                mag = pull_local_mag(data, abx, wf, so)
                #print(mag)
            else:
                mag = np.nan
            w_list.append(mag)
        mag_df[wf] = w_list
    return mag_df



def add_mag_sum(mag_df):
    """
        Add mag_sum column to df
        * added capacity to distinguish spin and spin_so with outputs mag_i for x,y,z
    """
    def collect_mag_vals(mag):
        """
            collects mag mom values for all lattice points and
            sums the values
        """
        mag_collect = []
        for key, val in mag.items():
            mag_collect.append(val)
        mag_collect = np.asarray(mag_collect)
        mag_sum = np.sum(mag_collect)
        return mag_sum

    wflist = ['initial','spin','afm','spin_so','afm_so']
    for wf in wflist:
        wf_vals = []
        for mag in mag_df[wf][:]:
            #print(mag, type(mag))
            if type(mag) == type([]):
                if 'so' in wf:
                    #print('wf',wf)
                    mag_xyz = []
                    for i in np.arange(3):
                        #print('i',i)
                        mag_i = mag[i]
                        #print('mag',mag_i)
                        mag_sum = collect_mag_vals(mag_i)
                        mag_xyz.append(mag_sum)
                    wf_vals.append(mag_xyz)
                else:
                    mag = mag[0]
                    #print(mag[0], type(mag))
                    mag_sum = collect_mag_vals(mag)
                    wf_vals.append(mag_sum)
            else:
                wf_vals.append(np.nan)
        wf_tot = wf + '_tot'
        mag_df[wf_tot] = wf_vals
    return mag_df
