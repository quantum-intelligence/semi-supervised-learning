# train_predict.py codes
from ml_methods import *



def train_and_predict(X2,t,KRRkernel):
    """codes to train and predict on data using algo"""
    split = 0.4
    cv_k = 8
    # Calculate best_gamma and best_lambda for KRR:
    r2_scores, best_lambda, best_gamma, Lam, gammas = get_krr_params(X2,t, KRRkernel, split, cv_k)
    featr = X2.columns
    #print(featr.shape)
    #print(featr)
    avg_te_pred, avg_tr_pred, Xkrr_train, Xkrr_test, ykrr_train,ykrr_test, scores, clfkrr, mae_info = krr_predict(X2, t,split,best_lambda, best_gamma)
    r2_krr_score = clfkrr.score(Xkrr_test, ykrr_test)
    print('KRR score with all features present : ', r2_krr_score)
    mae = np.mean(np.abs(avg_te_pred - ykrr_test))
    #print('mae', mae)
    return avg_te_pred, avg_tr_pred, ykrr_train, ykrr_test, r2_krr_score, mae




def plot_prediction(ykrr_test, avg_te_pred, ykrr_train, avg_tr_pred):
    """prediction performance plot"""
    plt.figure(figsize=(7.0,7.0))

    xx_min = np.min(avg_te_pred)*1.50
    xx_max = np.max(avg_te_pred)*1.50
    xx = np.arange(xx_min, xx_max+3.5)
    plt.plot(xx,xx,'--',c='gray',linewidth=1.5)

    print(ykrr_test.shape, type(ykrr_test))
    print(avg_te_pred.shape)
    plt.scatter(ykrr_test,avg_te_pred,marker='s',c='r',alpha=0.8,s=175,label='test')
    plt.scatter(ykrr_train,avg_tr_pred,marker='o',c='green',alpha=0.5,s=175,label='train')
    plt.xticks(fontsize=33)
    plt.yticks(fontsize=33)
    vmin = -9.5
    vmax = 0.5
    plt.xlim(vmin,vmax)
    plt.ylim(vmin,vmax)
    #plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()
    return


#avg_te_pred, avg_tr_pred, ykrr_train, ykrr_test, r2_krr_score, mae = train_predict(X2,t,KRRkernel)
#plot_prediction(ykrr_test, avg_te_pred, ykrr_train, avg_tr_pred)
