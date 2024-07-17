# codes for ml predictino of delta_FM_AFM
# prep scripts

def add_atom_counter(df):
    """
        add atom counts to datframe using
        get_unique_elem_info() and populating df columns
    """
    recalculate = True
    df_elements = get_unique_elem_info(df, recalculate=recalculate)
    df_elements.head()

    atom_label_list_spin, atom_count_list_spin = get_atom_counts(df, df_elements)
    df_counts = df.copy(deep = True)
    for ith, atom_label in enumerate(atom_label_list_spin):
        df_counts[atom_label] = atom_count_list_spin[ith]
    return df_counts

def filter_deltaE(df_fm_afm_so_Te, afm_fm_Te, fm_afm_Te, TMlist,B_atom_pair):
    """
        updates delta Energy(FM, AFM) using spin flip info from mag_metrics.ipynb
        - removes those values that have no well defined delta_E so can do regression analysis
    """
    # For AFM to FM:
    #
    indices = []
    for struc in afm_fm_Te:
        dex = np.where(struc == df_fm_afm_so_Te['formula'].values)[0][0]
        indices.extend(dex)
    df_fm_afm_so_Te2 = df_fm_afm_so_Te.copy()
    df_fm_afm_so_Te2 = add_atom_counter(df_fm_afm_so_Te2)
    #df_fm_afm_so_Te2['energy_dif'].loc[indices] = -11
    #df_fm_afm_so_Te2['spin_state'].loc[indices] = 1  # 1 for FM, 0 for AFM
    #df_fm_afm_so_Te2 = df_fm_afm_so_Te2.drop(df_fm_afm_so_Te2.index[indices])

    # For FM to AFM:
    #
    for struc in fm_afm_Te:
        dex = np.where(struc == df_fm_afm_so_Te['formula'].values)[0][0]
        indices.extend(dex)
    #df_fm_afm_so_Te3 = df_fm_afm_so_Te2.copy()
    #df_fm_afm_so_Te3 = add_atom_counts(df_fm_afm_so_Te3)
    #df_fm_afm_so_Te3['energy_dif'].loc[indices] = 11
    df_fm_afm_so_Te2 = df_fm_afm_so_Te2.drop(df_fm_afm_so_Te2.index[indices])

    return df_fm_afm_so_Te2




fm_afm = ['Ti1Cr1Si1Ge1Te6', 'Ti1Cr1Ge2Te6',  'Cr1Co1Si1Te6P1', 'Cr1Co1Ge1Te6P1', 'Cr1Co1Te6P2',
'Cr1Si1Ni1Te6P1','Cr1Ni1Ge2Te6','Cr1Ni1Ge1Te6P1','Cr1Ni1Te6P2',
'Cr1Cu1Si1Te6P1', 'Cr1Cu1Si2Te6', 'Cr1Cu1Ge2Te6', 'Cr1Cu1Ge1Te6P1', 'Cr1Cu1Si1Ge1Te6',
'Y1Cr1Si1Te6P1', 'Y1Cr1Ge1Te6P1',
'Ti1Cr1Si1Ge1Se6', 'Ti1Cr1Ge2Se6','Ti1Cr1P2Se6',
'V1Cr1Si1Ge1Se6', 'V1Cr1Ge2Se6', 'Cr2Si2Se6', 'Cr2Si1Ge1Se6', 'Cr2Ge2Se6',
'Mn1Cr1Si2Se6',  'Mn1Cr1Ge2Se6', 'Mn1Cr1Ge1P1Se6', 'Mn1Cr1P2Se6',
'Cr1Fe1Si1P1Se6', 'Cr1Co1Ge1P1Se6','Cr1Si1Ni1Ge1Se6','Cr1Ni1P2Se6',
'Cr1Si2Cu1Se6', 'Cr1Si1Cu1Ge1Se6', 'Cr1Si1Cu1P1Se6', 'Cr1Cu1Ge2Se6', 'Cr1Cu1Ge1P1Se6',
'Y1Cr1Si1Ge1Se6', 'Y1Cr1Ge2Se6', 'Nb1Cr1Si2Se6', 'Nb1Cr1Ge1P1Se6',
'Cr1Si2Ru1Se6','Cr1Si1Ge1Ru1Se6','Cr1Ge2Ru1Se6',
'Ti1Cr1Si2S6', 'Ti1Cr1Si1Ge1S6', 'Ti1Cr1Ge2S6',
'Cr1Ni1Ge2S6', 'Cr1Cu1Si1Ge1Te6','Cr1Cu1Ge2S6','Cr1Cu1Ge1P1S6', 'Y1Cr1Ge1P1S6','Y1Cr1P2S6',
'Cr1Ge2Ru1Se6']

afm_fm_flip_S = ['Y1Cr1Ge2S6', 'Nb1Cr1Si1Ge1S6', 'Cr1Co1Si1Ge1S6', 'Ti1Cr1Si1P1S6',
       'Y1Cr1Si2S6', 'Cr1Cu1Si2S6', 'Y1Cr1Si1Ge1S6', 'Cr1Co1Si2S6',
       'Nb1Cr1Ge2S6', 'Cr2Ge1P1S6', 'Ti1Cr1Ge1P1S6', 'Cr1Ge1P1Ru1S6',
       'Cr1Cu1P2S6', 'Cr1Co1Ge2S6', 'Cr2P2S6', 'Cr1Si1P1Ru1S6',
       'Y1Cr1Si1P1S6', 'Cr2Si1P1S6']

afm_fm_flip_Se = ['Cr1Co1Ge2Se6', 'Cr1Cu1P2Se6', 'Ti1Cr1Si1P1Se6', 'Y1Cr1Ge2Se6',
       'Ti1Cr1Ge1P1Se6', 'Cr1Co1Si1Ge1Se6', 'Cr1Co1P2Se6', 'Y1Cr1P2Se6',
       'Cr2Ge1P1Se6', 'Y1Cr1Si1Ge1Se6', 'Cr1Co1Si2Se6', 'Cr1Ge1P1Ru1Se6',
       'Y1Cr1Si2Se6', 'Cr1Si1P1Ru1Se6', 'Cr1Ge2Ru1Se6', 'Cr2Si1P1Se6']

afm_fm_flip_Te = ['Y1Cr1Si2Te6', 'Cr1Co1Si2Te6', 'Nb1Cr1Te6P2', 'Cr1Si1Te6P1Ru1',
       'Ti1Cr1Ge1Te6P1', 'Cr1Fe1Te6P2', 'Y1Cr1Si1Ge1Te6', 'Cr1Cu1Te6P2',
       'Cr1Ge2Te6Ru1', 'Nb1Cr1Si1Te6P1', 'Cr1Fe1Si1Te6P1', 'Cr1Si2Te6Ru1',
       'Y1Cr1Ge2Te6', 'Cr1Co1Ge2Te6', 'Y1Cr1Te6P2', 'Cr2Si1Te6P1',
       'Cr1Co1Si1Ge1Te6', 'Cr1Te6P2Ru1', 'Cr1Fe1Ge1Te6P1']

afm_fm = []
afm_fm.extend(afm_fm_flip_Te)
afm_fm.extend(afm_fm_flip_Se)
afm_fm.extend(afm_fm_flip_S)


# df_optima_counts_nanp
print(df_optima_counts_nan.shape)
df_fm_afm_so3 = filter_deltaE(df_optima_counts_nan, afm_fm, fm_afm, TMlist,B_atom_pair)
print(df_fm_afm_so3.shape)

target_label = 'delta_FM_AFM'
dataname = 'delta_FM_AFM_df'
X_mag, t_mag, formula_mag, df_mag_original  = df_to_X(df_fm_afm_so3, target_label, is_target,dataname, recalculate=True)

####


max_depth = 30
n_estimators = 25 # should be as large as procssor will allow
min_samples_leaf = 2
max_features = 'auto'
testsplit = 0.2
rstate = 1
Xmag_train, Xmag_test, ymag_train, ymag_test = scaledata(X_mag, t_mag, testsplit, rstate)

from sklearn.model_selection import cross_val_score
# >>> clf = svm.SVC(kernel='linear', C=1)
# >>> scores = cross_val_score(clf, iris.data, iris.target, cv=5)

def erf_fit_cv(X_train,y_train,X_test,y_test,max_depth,max_features,min_samples_leaf,n_estimators):
    """
       Function packaging fitting routine for random forests, passing in dadta and hyperparameters and spitting out
       MSE for those hyperparameters
       returns: prediton for test data, y_erg; predictino for training dta yt_erf, mean-square-error and absolute-error
    """
    #regr_erf = ExtraTreesRegressor(max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,
    #                                n_estimators=n_estimators,random_state=2,n_jobs = -1)
    rand_seed = 3
    regr_erf = ExtraTreesRegressor(n_estimators=n_estimators, criterion='mse', max_depth=max_depth,
                                   min_samples_split=2,
                                   min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0,
                                   max_features=max_features,
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                   bootstrap=False, oob_score=False, n_jobs=1, random_state=rand_seed, verbose=0,
                                   warm_start=False)
    regr_erf.fit(X_train, y_train)
    y_erf = regr_erf.predict(X_test)
    yt_erf = regr_erf.predict(X_train)
    mse, avg_dif = get_mse(y_test,y_erf)
    cvscores = cross_val_score(regr_erf, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    #TAKE MEAN of CVscores
    cvscores = np.mean(np.abs(cvscores))
    return y_erf, yt_erf, cvscores, mse, avg_dif, regr_erf

######

#
min_samples_leaf = 3
n_jobs = -1
max_features = 'auto' #must be < 28 ie less than total number of descriptors #'auto'
# A smaller leaf makes the model more prone to capturing noise in train data.
# prefer a minimum leaf size of more than 50

depth = np.arange(1,20)*1
estimators  = np.arange(1,20)*1
features = np.linspace(1,100,10)

rf_acc2_list = []

for d in tqdm(depth):
    rf_acc_list = []
    for est in estimators:
        ymag_erf,ymagt_erf,cvscores, mse2,avg_dif, regr2_erf = erf_fit_cv(Xmag_train,
                                                     ymag_train,Xmag_test,ymag_test,d,
                                                     max_features,min_samples_leaf,est)
        #mse2 is implemented as CV score
        rf_acc_list.append(cvscores)
        #print(cvscores.shape)
    rf_acc2_list.append(rf_acc_list)
rf_acc_array = np.asarray(rf_acc2_list)


#######



print(rf_acc_array.shape)
plt.imshow(rf_acc_array)
plt.colorbar()


######


print(rf_acc_array.shape)
minindex = np.argwhere(rf_acc_array.min()==rf_acc_array)[0]
print(minindex)
est_min = estimators[minindex[1]]
d_min = depth[minindex[0]]
print('best # estimator:', est_min)
print('best tree depth:', d_min)
print(rf_acc_array[minindex[0],minindex[1]])

######

max_depth= d_min
n_estimators = est_min #should be as large as procssor will allow
n_jobs = -1
max_features = 'auto' #Max features is als an important tuning parameter!

np.random.seed(12)

iterations = 50
scores = []
std_var = []
msevals = []
testscores = []
for iter in tqdm(np.arange(iterations)):   #mmm
    # A smaller leaf makes the model more prone to capturing noise in train data.
    # prefer a minimum leaf size of more than 50
    rstate = np.random.randint(0,10)+110
    Xmag_train, Xmag_test, ymag_train, ymag_test = scaledata(X_mag,t_mag,testsplit, rstate)
    ymag_erf,ymagt_erf,cvscores, mse2,mae_test, regr_erf2 = erf_fit_cv(Xmag_train,ymag_train,Xmag_test,ymag_test,max_depth,
                                                    max_features,min_samples_leaf,n_estimators)
    #print(cvscores)
    #scores.append(np.mean(cvscores))
    mse_train, mae = get_mse(ymag_train,ymagt_erf)
    scores.append(mae)
    #std_var.append(np.var(cvscores))
    msevals.append(mse2)

    mae_test = np.mean(np.abs( regr_erf2.predict(Xmag_test) - ymag_test))
    testscores.append(mae_test)
    #testscores.append(regr_erf2.score(Xmag_test,ymag_test))
    #print('MSE for the random forets is: ', mse2, 'avg dif', avg_dif)
    #print('extra forest R^2', np.round( regr_erf2.score(Xmag_test,ymag_test),3) )

meanscore =  np.mean(testscores)
varscore = np.std(testscores)
print('********* \nAverage testscore (MAE):', "%.3f"% meanscore, ' +/- ', "%.3f"% varscore)
print('\n*********')


#####

plt.figure(figsize=(6,6))
plt.scatter(np.arange(len(scores)),scores,marker='o',color='green',alpha=0.5)
# plt.scatter(np.arange(len(scores)),std_var,marker='x',color='blue')
plt.scatter(np.arange(len(scores)),testscores,marker='p',color='red',s=80)
plt.show()


####


%matplotlib inline
plt.figure()
s = 25
a = 0.60
vmin = np.min(ymag_train)*0.10
vmax = np.max(ymag_train)*1.15
plt.figure(figsize=(9,9))
plt.scatter(ymag_test, ymag_erf, c="red", s=170, marker="s", alpha=0.85, label="test")
plt.scatter(ymag_train, ymagt_erf, c="green", s=150, marker="o", alpha=a, label="training")
xline=np.linspace(vmin,vmax,50)
plt.plot(xline,xline,'--',c='gray', linewidth=2)
# plt.xlim([-3, 1.0])
# plt.ylim([-3, 1.0])
plt.xlabel("DFT $\mu$ [$\mu_B$]",fontsize=35)
plt.ylabel("Predicted $\mu$ [$\mu_B$]",fontsize=35)
plt.title("Extra forest $\mu$ prediction, X=Te \n",fontsize=22)
#plt.title("Extra forest $\mu$ prediction, X=Te,Se,S [cutoff = 0.07]\n",fontsize=22)
#plt.legend(loc=0, prop={'size':19})
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(False)
# plt.xlim(vmin,vmax)
# plt.ylim(vmin,vmax)
plt.xlim(-0.0,8)
plt.ylim(-0.0,8)
#plt.legend()
plt.show()




%matplotlib inline
krr_features = X_mag.columns
print(len(krr_features))
importance = regr2_erf.feature_importances_
plt.figure(figsize=(7,7))
krr_features = np.array(krr_features)
importance_index = np.argsort(importance)
erf_sort_features = krr_features[importance_index]
vip = importance[importance_index]
vip_matrix = [importance, krr_features]
bar_width = 0.35
max_show = 10
index= np.arange(len(krr_features[-max_show:]))
plt.bar(np.arange(len(importance[-max_show:])),vip[-max_show:])
#plt.xticks(index + bar_width, erf_sort_features[-max_show:],rotation='48',fontsize=26,ha='right')

mylabels = erf_sort_features[-max_show:]

plt.xticks(index + bar_width, mylabels,rotation='48',fontsize=25,ha='right')
plt.ylabel('Descriptor importances',fontsize=20)
plt.yticks(fontsize=30)
tree_accuracy = regr_erf2.score(Xmag_test,ymag_test)

erf_sort_features[::-1]
# erf_sort_features[:]
