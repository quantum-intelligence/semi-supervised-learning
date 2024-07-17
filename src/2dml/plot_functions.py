# plot functions
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from alloy_functions import *


def plot_mag(spinMatrix_mag,title,B_atom_pair,TMlist,vmin,vmax):
    """ plots magnetic moments from spinMAtrix """
    plt.figure(figsize=(6,6))

    cmap='Blues'
    current_cmap = matplotlib.cm.get_cmap('Purples')
    current_cmap.set_bad(color='grey')

    # plt.imshow(spinMatrix_dif, cmap='inferno')
    plt.imshow(spinMatrix_mag, cmap=current_cmap,vmin=vmin,vmax=vmax)

    # plt.colorbar(label='magnetic moment / unit cell')
    Batoms = [' '.join(x) for x in B_atom_pair]

    font_size = 20 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)

    x = np.arange(spinMatrix_mag.shape[1])
    y = np.arange(spinMatrix_mag.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.grid(False)
    plt.title(title)
    plt.xticks(x, xlabels,rotation='vertical',fontsize=20)
    plt.yticks(y, ylabels,rotation='horizontal',fontsize=20)
    plt.show()




def plot_elem_mag(df_counts,elem,label,nbins):
    binwidth = 0.550
    mag_other = np.abs(df_counts[label][df_counts[elem] == 0].values)
    binsinfo =np.arange(min(mag_other),max(mag_other)+binwidth,binwidth)
    plt.hist(mag_other, color='g', bins=binsinfo, alpha=0.6,density=True, label=elem)
    mag_target = np.abs(df_counts[label][df_counts[elem] >= 1].values)
    #binsinfo =np.arange(min(mag_target),max(mag_target)+binwidth,binwidth)
    plt.hist(mag_target, color='r', bins=binsinfo, alpha=0.6,density=True, label=elem)
    #plt.xlabel('magnetic moment',fontsize=30)
    #plt.ylabel('counts',fontsize=30)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.ylim(0,1.1)
    #plt.legend(fontsize=20)



def plot_elem(df_counts,elem,label,nbins):
    #
    binwidth = 0.5
    e_other = df_counts[label][df_counts[elem] == 0].values
    binsinfo =np.arange(np.min(e_other),np.max(e_other)+binwidth,binwidth)
    #
    df_counts[label][df_counts[elem] == 0].hist(color='g', bins=binsinfo, alpha=0.6,normed=True)
    df_counts[label][df_counts[elem] >= 1].hist(color='r', bins=binsinfo, alpha=0.6,normed=True)
    #plt.xlabel('cohesive energy',fontsize=30)
    #plt.ylabel('counts',fontsize=30)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    plt.grid(False)
    plt.locator_params(nbins=4,axis='y')
    plt.ylim(0,0.65)
    plt.xlim(-8.5,1.5)
    plt.legend()


#     ###  ------ COPY ------  ###
# def plot_elem(df_counts,elem,label,nbins):
#     #
#     binwidth = 0.5
#     e_other = df_counts[label][df_counts[elem] == 0].values
#     #binsinfo =np.arange(np.min(e_other),np.max(e_other)+binwidth,binwidth)
#     #
#     df_counts[label][df_counts[elem] == 0].hist(color='g', bins=20, alpha=0.6,normed=True)
#     df_counts[label][df_counts[elem] >= 1].hist(color='r', bins=20, alpha=0.6,normed=True)
#     plt.xlabel('cohesive energy',fontsize=30)
#     plt.ylabel('counts',fontsize=30)
#     plt.xticks(fontsize=35)
#     plt.yticks(fontsize=35)
#     plt.grid(False)
#     plt.locator_params(nbins=4,axis='y')
# #     plt.ylim(0,0.65)
# #     plt.xlim(-8.5,1.5)
# #     plt.legend()



def energyplot(spinMatrix_dif, B_atom_pair, TMlist, title, cmaplabel,vmin,vmax,range=True):
    """
        constructe 2D energy difference plot using Matrix input
    """
    #plt.figure(figsize=(6,6))

    current_cmap = matplotlib.cm.get_cmap(name=cmaplabel)
    current_cmap.set_bad(color='grey')
    if range == True:
        plt.imshow(spinMatrix_dif, cmap=current_cmap,vmin=vmin,vmax=vmax)
    else:
        plt.imshow(spinMatrix_dif, cmap=current_cmap)

    Batoms = [' '.join(x) for x in B_atom_pair]
    # print(len(Batoms), Batoms)
    # print(spinMatrix_nna.shape)
    x = np.arange(spinMatrix_dif.shape[1])
    y = np.arange(spinMatrix_dif.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.title(title)
    plt.grid(False)
    plt.xticks(x, xlabels,rotation='vertical', fontsize=35)
    plt.yticks(y, ylabels,rotation='horizontal', fontsize=35)
    font_size = 35 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)

    # decrease number of ticks
    tick_locator = matplotlib.ticker.MaxNLocator(nbins=5)
    cb.locator = tick_locator
    cb.update_ticks()
    #plt.show()
    return


def EnergyMatrixGen(df_counts,TMlist,B_atom_pair, label):
    """
        generates EnergyMAtrix from B list, TM list
        constrain for Te containing, or Se containing atoms etc prior
        - label -> 'cohesive', 'magmom' etc.
        -1/10: corrected error with Cr atom always being found and index being updated erroneously..
    """
    df_counts1 = df_counts.copy(deep=True)
    if 'level_0' in df_counts1.columns:
        df_counts1 = df_counts1.drop(columns='level_0')
    if 'elem_frac_edf' in df_counts1.columns:
        df_counts1 = df_counts1.rename(columns={'elem_frac_edf':'elem_frac'})
    df_counts1 = df_counts1.reset_index()
    EnergyMatrix = np.empty((len(TMlist),len(B_atom_pair)))
    print(EnergyMatrix.shape)
    if 'formula_edf' in df_counts.columns:
        df_counts1['formula'] = df_counts1['formula_edf']
    for i,cmpd in enumerate(df_counts1['formula'][:]):
        for bth, b in enumerate(B_atom_pair):
            for tmth, tm in enumerate(TMlist):
                if tm == 'Cr': #special case when Cr is tm. always will find it ipresent...
                    TMtrue = df_counts1.loc[i,'Cr'] == 2.0
                    #print(TMtrue)
                    Btrue = Bexists(i, b, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_energy = df_counts1['cohesive'][i]
                        TMB_energy = df_counts1[label][i]
                        #print(tm, b, TMB_energy)
                        EnergyMatrix[tmth,bth] = TMB_energy
                else:
                    #print(i,bth,tmth)
                    Btrue = Bexists(i, b, df_counts1)
                    TMtrue = TMexists(i, tm, df_counts1)
                    if Btrue and TMtrue:
                        #TMB_energy = df_counts1['cohesive'][i]
                        TMB_energy = df_counts1[label][i]
                        EnergyMatrix[tmth,bth] = TMB_energy
    EnergyMatrix_nna = EnergyMatrix.copy()
    EnergyMatrix_nna[np.isnan(EnergyMatrix_nna)] = np.nan
    return EnergyMatrix_nna





def energyscape(df_counts,X,feature_label,TMlist,B_atom_pair,vmin,vmax):
    """
        creates cohesive energy 2D plot
    """
    df_counts_Te = df_counts[df_counts[X] ==6].copy()
    df_counts_Te_cp = df_counts_Te.copy()
    #feature_label = 'cohesive'
    EnergyMatrix_nna = EnergyMatrixGen(df_counts_Te_cp,TMlist,B_atom_pair,feature_label)

    plt.figure(figsize=(12,8))
    #cmap='spectral'
    current_cmap = matplotlib.cm.get_cmap('coolwarm')
    current_cmap.set_bad(color='white')
    #vmin = -4
    #vmax = 2.0
    plt.imshow(EnergyMatrix_nna,cmap=current_cmap, interpolation='none',vmin=vmin,vmax=vmax)

    #plt.colorbar()
    font_size = 20 # Adjust as appropriate.
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=font_size)

    Batoms = [' '.join(x) for x in B_atom_pair]
    print(len(Batoms), Batoms)
    print(EnergyMatrix_nna.shape)
    x = np.arange(EnergyMatrix_nna.shape[1])
    y = np.arange(EnergyMatrix_nna.shape[0])
    xlabels = Batoms
    ylabels = TMlist
    plt.grid(False)
    #plt.title(' TEST predition')
    plt.xticks(x, xlabels,rotation='vertical',fontsize=20)
    plt.yticks(y, ylabels,rotation='horizontal',fontsize=20)
    plt.show()
    return EnergyMatrix_nna



def cohesive_matrix_plot(df_spin_Te, df_elements,TMlist, B_atom_pair,title ):
    # Create atom counts:
    atom_label_list_spin, atom_count_list_spin = get_atom_counts(df_spin_Te, df_elements)
    df_spin_counts_Te = df_spin_Te.copy(deep = True)
    for ith, atom_label in enumerate(atom_label_list_spin):
        #print(atom_label)
        df_spin_counts_Te[atom_label] = atom_count_list_spin[ith]
        #atom_count_list

    # print(B_atom_pair)
    descriptor = 'cohesive'
    spinMatrix_Te = spinMatrixGen(df_spin_counts_Te,TMlist,B_atom_pair,descriptor)

    # Create plot

    cmaplabel = 'inferno'
    plt.figure(figsize=(8.5,8.5))
    vmin = -8; vmax=0.5;
    plf.energyplot(spinMatrix_Te, title, cmaplabel,vmin,vmax,range=True)
    #plt.colorbar()
    return
