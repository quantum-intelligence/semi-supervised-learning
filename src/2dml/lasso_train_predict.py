# lasso_train_predict

from import_functions import *

def calc_hyperparameter(X2_train, X2_test, y2_train, y2_test):
    """ calculate LASSO hyperparameter """
    Lam = np.arange(-10,10)*0.66
    Lam = 10.0**Lam
    best_lam = 0
    best_score = 0
    scores = np.zeros((len(Lam),1))
    score = -1.0
    for ith, lam in enumerate(Lam):
        reg = linear_model.Lasso(alpha = lam)
        reg.fit(X2_train, y2_train)
        yl_pred = reg.predict(X2_test)
        # This scoring method does not use clever cross-validation
        #lasso_score = reg.score(X2_test, y2_test)
        #
        xdata = np.concatenate((X2_train, X2_test))
        ydata = np.concatenate((y2_train, y2_test))
        cv_scores = cross_val_score(reg, X2_train, y2_train, cv=5)
        #cv_scores = cross_val_score(reg, X2_test, y2_test, cv=3)
        lasso_score = np.mean(cv_scores)
        #
        if score < lasso_score:
            score = lasso_score
            best_pred = yl_pred
            best_lam = lam
            best_score = score
            #best_test = yl_test
        scores[ith] = lasso_score
    # plt.plot(Lam,scores,'o-')
    # plt.xlabel('lambda',fontsize=15)
    # plt.ylabel('score ',fontsize=15)
    # plt.xscale('log')
    # print('labda: ', best_lam, 'with score', best_score)
    return best_lam

##############################################################

def lasso_predict(X2_train, X2_test, y2_train, y2_test, best_lam):
    """ make lasso prediction """
    reg = linear_model.Lasso(alpha = best_lam)
    reg.fit(X2_train, y2_train)
    lasso_tr_score = reg.score(X2_train,y2_train)
    lasso_te_score = reg.score(X2_test,y2_test)
    print('train score: ', lasso_tr_score, '; test score: ', lasso_te_score)
    yl_pred = reg.predict(X2_test)
    ylt_pred = reg.predict(X2_train)
    return yl_pred, ylt_pred

###############################################################

def lasso_plot_perform(y2_test,yl_pred,y2_train,ylt_pred):
    """ plot model performance for lasso """
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    ymax = np.max(yl_pred)
    minv = np.min(y2_train)
    maxv = np.max(y2_train)
    xx = np.arange(minv*1.5,maxv*1.5)
    plt.plot(xx,xx,'--',color='cyan')
    #plt.xlim(0.2,701)
    #plt.ylim(0.2,701)
    plt.plot(y2_test,yl_pred,'s', markersize=9,alpha=0.8)
    plt.plot(y2_train,ylt_pred,'.', markersize=9,alpha=0.8)
    plt.xlabel('actual cohesive energy', fontsize=20)
    plt.ylabel('predicted cohesive energy', fontsize=20)
    plt.show()
