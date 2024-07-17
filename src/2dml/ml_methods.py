# ml_methods.py

import os

import pymatgen as mg
import pymatgen as mp


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
#mpr = MPRester("0Eq4tbAMNRe37DPs")
from pymatgen.ext.matproj import MPRester
m = MPRester("RK5GrTk1anSOmgAU")

from scipy.stats import skew
import shutil

from fmmlcalc_b import *
from atomic_property import *
from tqdm import tqdm

from sklearn.model_selection import KFold

# import tensorflow as tf
from sklearn.ensemble import ExtraTreesRegressor


def kfold_score(X2, y, numfolds, KRRkernel, best_gamma, best_lambda):
    """
        modified lines:    test_predictions.append(krrte_pred[:])
        train_predictions.append(krrtr_pred[:])
    """
    clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
    y2np = np.asarray(y.values)
    #X2np = np.asarray(X2.drop('p_tanimoto',1))
    snpdata2, snptarget2 = scaledata_xy(X2,y)
    X2np = snpdata2
    kf = KFold(n_splits=numfolds)
    guess_train_size = (len(snpdata2)/numfolds)
    guess_test_size = (len(snptarget2)/numfolds)
    #print guess_train_size, guess_test_size
    testscores = []
    trainscores = []
    test_predictions = []
    train_predictions = []
    for train_index, test_index in kf.split(X2np):
        # print("TRAIN:", train_index, "TEST:", test_index)
        Xk_train, Xk_test = X2np[train_index], X2np[test_index]
        yk_train, yk_test = y2np[train_index], y2np[test_index]

        clfkrr.fit(Xk_train, yk_train)
        testscore = clfkrr.score(Xk_test,yk_test)
        trainscore = clfkrr.score(Xk_train,yk_train)
        #print testscore , trainscore
        testscores.append(testscore)
        trainscores.append(trainscore)
        krrte_pred = clfkrr.predict(Xk_test)
        krrtr_pred = clfkrr.predict(Xk_train)
        #print('guess_test_size, guess_train_size', guess_test_size, guess_train_size)
        #print('krrte_pred', krrte_pred, 'krrte_pred', krrtr_pred)
        #test_predictions.append(krrte_pred[:guess_test_size])
        #train_predictions.append(krrtr_pred[:guess_train_size])
        test_predictions.append(krrte_pred[:])
        train_predictions.append(krrtr_pred[:])

    meantestscore = np.mean(testscores)
    meantrainscore = np.mean(trainscores)
    pred_len = len(test_predictions)-2
    test_pred_len = len(train_predictions)-2
    #print len(test_predictions), pred_len, test_pred_len
    #print test_predictions
    mean_testp = np.mean(test_predictions[:pred_len],axis=0)
    mean_trainp = np.mean(train_predictions[:test_pred_len],axis=0)
    return meantestscore, meantrainscore, mean_testp, mean_trainp



def krr_predict(X2,target,split,best_lambda, best_gamma):
    """
        calc krr predictions
        - added best_lambda and bsst_gamma as inputs 4-24-2019.
        ** Otherwise how is getting access to these ablues?

    """
    # Use krr model to generate predictinos for a set of data
    np.random.seed(12)
    snpdatakrr, snptargetkrr = scaledata_xy(X2,target)
    #runs = 1 #cant do more than one run if youre randomly splitting test and train set each time!!
    # te_preds = []
    # tr_preds = []
    # te_mae_list = []
    #for nth in np.arange(runs):
    nth_state = np.random.randint(1,50)
    #Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = cross_validation.train_test_split(snpdatakrr, snptargetkrr, test_size=split, random_state=nth_state)
    Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = train_test_split(snpdatakrr, snptargetkrr, test_size=split, random_state=nth_state)
    clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
    clfkrr.fit(Xkrr_train, ykrr_train)
    krrte_pred = clfkrr.predict(Xkrr_test)
    krrtr_pred = clfkrr.predict(Xkrr_train)
    tr_r2 = clfkrr.score(Xkrr_train, ykrr_train)
    te_r2 = clfkrr.score(Xkrr_test, ykrr_test)
    #te_preds.append(krrte_pred)
    #tr_preds.append(krrtr_pred)
    mae = np.mean(np.abs(ykrr_test - krrte_pred))
    # te_mae_list.append(mae)
    #
    # avg_te_pred = np.mean(te_preds,axis=0)
    # avg_tr_pred = np.mean(tr_preds,axis=0)
    # te_mae = np.mean(te_mae_list)
    # mae_std = np.std(te_mae_list)
    scores = [tr_r2, te_r2]
    # mae_info = [te_mae, mae_std]
    # return avg_te_pred, avg_tr_pred, Xkrr_train, Xkrr_test, ykrr_train, ykrr_test, scores, clfkrr, mae_info
    return krrte_pred, krrtr_pred, Xkrr_train, Xkrr_test, ykrr_train, ykrr_test, scores, clfkrr, mae



def krr_predict_stats(X2,target,split,best_lambda, best_gamma, runs):
    """
        calc krr predictions
        - added best_lambda and bsst_gamma as inputs 4-24-2019.
        ** Otherwise how is getting access to these ablues?

    """
    # Use krr model to generate predictinos for a set of data
    snpdatakrr, snptargetkrr = scaledata_xy(X2,target)
    # runs = 1 #cant do more than one run if youre randomly splitting test and train set each time!!
    te_preds = []
    tr_preds = []
    te_mae_list = []
    for nth in np.arange(runs):
        nth_state = np.random.randint(1,50)
        #Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = cross_validation.train_test_split(snpdatakrr, snptargetkrr, test_size=split, random_state=nth_state)
        Xkrr_train, Xkrr_test, ykrr_train, ykrr_test = train_test_split(snpdatakrr, snptargetkrr, test_size=split, random_state=nth_state)
        clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
        clfkrr.fit(Xkrr_train, ykrr_train)
        krrte_pred = clfkrr.predict(Xkrr_test)
        krrtr_pred = clfkrr.predict(Xkrr_train)
        tr_r2 = clfkrr.score(Xkrr_train, ykrr_train)
        te_r2 = clfkrr.score(Xkrr_test, ykrr_test)
        te_preds.append(krrte_pred)
        tr_preds.append(krrtr_pred)
        mae = np.mean(np.abs(ykrr_test - krrte_pred))
        te_mae_list.append(mae)
    # avg_te_pred = np.mean(te_preds,axis=0)  # This is the average of randomly chosen training/test splits
    # avg_tr_pred = np.mean(tr_preds,axis=0)
    te_mae = np.mean(te_mae_list)
    mae_std = np.std(te_mae_list)
    scores = [tr_r2, te_r2]
    mae_info = [te_mae, mae_std]
    # return avg_te_pred, avg_tr_pred, Xkrr_train, Xkrr_test, ykrr_train, ykrr_test, scores, clfkrr, mae_info
    return  Xkrr_train, Xkrr_test, ykrr_train, ykrr_test, scores, clfkrr, mae_info



def krr_TEST_predict(X_TEST,clfkrr):
    """
        calc krr estimates on X_test using trained model
        - based on krr_predict()
	found in ml_methods.py
    """
    # Use krr model to generate predictinos for a set of data
    dummy_target = np.ones(len(X_TEST))
    dummy_target = pd.DataFrame(dummy_target)
    snpdatakrr, snptargetkrr = scaledata_xy(X_TEST,dummy_target)
    #clfkrr = KernelRidge(kernel=KRRkernel, gamma = best_gamma, alpha = best_lambda)
    #clfkrr.fit(Xkrr_train, ykrr_train)
    te_pred = clfkrr.predict(snpdatakrr)
    return te_pred





def get_mse(y_test,prediction):
    """
        Calculates the Mean Squared Error of test data and predictions
        """
    acc = np.mean((y_test-prediction)**2.0);
    mean_dif = np.mean(np.abs(y_test-prediction))
    return acc, mean_dif



def get_krr_params(X2, target, KRRkernel, split, cv_k):
    """ determine hyperparameters for KRR """
    #KRRkernel = 'rbf'
    #X2.columns
    #snpdata_krr, snptarget_krr = scaledata_xy(X_krr,y2_target)
    snpdata, snptarget = scaledata_xy(X2,target)
    #X2_train, X2_test, y2_train, y2_test = scaledata(X2,target,0.25,rstate=0)
    Lam = np.linspace(-18, 1, num=30)*0.4
    Lam = 10.0**Lam
    gammas = np.linspace(-20,30,num=30)*0.5
    gammas = 10.0**gammas
    r2_scores = np.zeros((len(Lam),len(gammas)))
    best_lambda = 0
    best_gamma = 0
    best_score = -10
    Nrun = 1
    for llth, ll in tqdm(enumerate(Lam)):
        for ggth, gg in enumerate(gammas):
            score_runs = []
            for run in np.arange(Nrun):
                clfkrr = KernelRidge(kernel=KRRkernel,gamma = gg, alpha=ll)
                #use k-fold cross_validation
                #r2krr = cross_val_score(clfkrr, snpdata2_para, snptarget2_para, cv=8)
                #r2krr = np.mean(r2krr)
                #rstate = np.random.randint(1,10)
                rstate = 20
                #split = 0.4
                X2_train, X2_test, y2_train, y2_test = scaledata(X2,target,split,rstate)
                clfkrr.fit(X2_train, y2_train)
                #r2krr = clfkrr.score(Xkrr_test,ykrr_test) #Using only 25% of data for score wiht no CV is
                #                                         # dangerous!
                # r2krr = cross_val_score(clfkrr, snpdata, snptarget, cv=5) # do CV on entire dataset.. overfit?
                # r2krr = cross_val_score(clfkrr, X2_train, y2_train, cv=3) # this overfits to the training data
                r2krr = cross_val_score(clfkrr, X2_test, y2_test, cv=3) # why would you do this?!!
                r2krr = cross_val_score(clfkrr, X2_train, y2_train, cv=cv_k)
                r2krr = np.mean(r2krr)
                score_runs.append(r2krr)
            score_runs_mean = np.mean(np.asarray(score_runs))
            #print score_runs_mean
            if best_score < score_runs_mean:
                best_score = r2krr
                best_lambda = ll
                best_gamma = gg
            # r2_scores[llth,ggth] = r2krr #had this before... 9/21/2018
            r2_scores[llth,ggth] = score_runs_mean

    print(best_score)
    print('best_lambda', best_lambda)
    print('best_gamma', best_gamma)
    return r2_scores, best_lambda, best_gamma, Lam, gammas
#plt.semilogx(Lam,r2_scores,'-o')

# Copied frrom jupyter notebook... sure is not different from version above??
# def get_krr_params(X2, target, KRRkernel, split, cv_k):
#     """ determine hyperparameters for KRR """
#     #KRRkernel = 'rbf'
#     #X2.columns
#     #snpdata_krr, snptarget_krr = scaledata_xy(X_krr,y2_target)
#     snpdata, snptarget = scaledata_xy(X2,target)
#     #X2_train, X2_test, y2_train, y2_test = scaledata(X2,target,0.25,rstate=0)
#     Lam = np.linspace(-9, 9, num=30)*0.4
#     Lam = 10.0**Lam
#     gammas = np.linspace(-20,20,num=30)*0.4
#     gammas = 10.0**gammas
#     r2_scores = np.zeros((len(Lam),len(gammas)))
#     best_lambda = 0
#     best_gamma = 0
#     best_score = -10
#     Nrun = 1
#     for llth, ll in tqdm(enumerate(Lam)):
#         for ggth, gg in enumerate(gammas):
#             score_runs = []
#             for run in np.arange(Nrun):
#                 clfkrr = KernelRidge(kernel=KRRkernel,gamma = gg, alpha=ll)
#                 # use k-fold cross_validation:
#                 rstate = 20
#                 X2_train, X2_test, y2_train, y2_test = scaledata(X2,target,split,rstate)
#                 # clfkrr.fit(X2_train, y2_train)
#                 # r2krr = cross_val_score(clfkrr, X2_train, y2_train, cv=3) # this overfits to the training data
#                 # r2krr = cross_val_score(clfkrr, X2_test, y2_test, cv=3) # why would you do this?!!
#                 r2krr = cross_val_score(clfkrr, X2_train, y2_train, cv=cv_k)
#                 r2krr = np.mean(r2krr)
#                 score_runs.append(r2krr)
#             score_runs_mean = np.mean(np.asarray(score_runs))
#             if best_score < score_runs_mean:
#                 best_score = r2krr
#                 best_lambda = ll
#                 best_gamma = gg
#             # r2_scores[llth,ggth] = r2krr #had this before 9/21/2018
#             r2_scores[llth,ggth] = score_runs_mean
#     print(best_score)
#     print('best_lambda', best_lambda)
#     print('best_gamma', best_gamma)
#     return r2_scores, best_lambda, best_gamma, Lam, gammas
# #plt.semilogx(Lam,r2_scores,'-o')





def erf_fit(X_train,y_train,X_test,y_test,max_depth,max_features,min_samples_leaf,n_estimators):
    """
        Function packaging fitting routine for random forests, passing in dadta and hyperparameters and spitting out
        MSE for those hyperparameters
        returns: prediton for test data, y_erg; predictino for training dta yt_erf, mean-square-error and absolute-error
        """
    #regr_erf = ExtraTreesRegressor(max_depth=max_depth,max_features=max_features,min_samples_leaf=min_samples_leaf,
    #                                n_estimators=n_estimators,random_state=2,n_jobs = -1)
    regr_erf = ExtraTreesRegressor(n_estimators=n_estimators, criterion='mse', max_depth=max_depth,
                                   min_samples_split=2,
                                   min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=0.0,
                                   max_features=max_features,
                                   max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None,
                                   bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0,
                                   warm_start=False)
    #print 'xtrainshape', X_train.shape
    #print 'ytrainsahpe',y_train.shape
    regr_erf.fit(X_train, y_train)
    #print 'got here'
    # Predi ct on new data
    y_erf = regr_erf.predict(X_test)
    yt_erf = regr_erf.predict(X_train)
    mse, avg_dif = get_mse(y_test,y_erf)
    return y_erf, yt_erf, mse, avg_dif, regr_erf



def erf_eval_performance(d_min,est_min,df1_nna,splits,X2,t ):
    """ get test perofrmance of extra forest regression"""
    max_depth= d_min
    n_estimators = est_min #should be as large as procssor will allow
    min_samples_leaf = 2
    n_jobs = -1
    max_features = 'auto' #Max features is als an important tuning parameter!
    target_label = 'cohesive'
    # REPLACE X2_gen with df_to_X incorporates hardness and other features
    # X2, t, formula = X2_gen(df1_nna,target_label,is_target=True)
    #
    # CALC X2, t inside each split is to expensive nad not needed.

    #splits = (np.arange(9)+1.0)*0.1 #defines test set size..
    #splits = splits[::-1]
    #print('splits :  ', splits)
    mse_vals = []
    ase_vals = []
    r2_vals = []
    for split in splits:
        #print(split)
        rstate = random.randint(1,10)
        X2_train, X2_test, y2_train, y2_test = scaledata(X2,t ,split,rstate)
        #print('X2train shape',X2_train.shape)
        y2_erf,y2t_erf,mse2,avg_dif, regr_erf2 = erf_fit(X2_train,y2_train,X2_test,y2_test,max_depth,
                                                         max_features,min_samples_leaf,n_estimators)
        mse_vals.append(mse2)
        ase_vals.append(avg_dif)
        r2_vals.append(np.round( regr_erf2.score(X2_test,y2_test),3))
        #print('MSE for the random forets is: ', mse2, 'avg dif', avg_dif)
        #print('extra forest R^2',  )
        train_splits = 1-splits
    return train_splits,mse_vals,ase_vals,r2_vals
