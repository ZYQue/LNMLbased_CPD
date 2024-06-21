import sys
import numpy as np
import os.path
import pandas as pd
import math
from scipy import linalg
from sklearn.covariance import graphical_lasso
import matplotlib.pyplot as plt
from glassobind import glasso
from sklearn.covariance import GraphicalLasso
import statsmodels.stats.correlation_tools as stat_tools

def data_imputation(data_ori,data_filled):
    # input: original time series data with missing values and data filled by N(0,1)
    # output: loss

    with_CV=True
    eps=1.0e-3
    n=np.size(data_ori,0)	# length of time series
    p=np.size(data_ori,1)	# dimensionality
    print("Start data imputation: length "+str(n)+"\t dim "+str(p))

    # data imputation method: Loh-Wainright
    M=np.zeros((p,p))
    obs_prob=np.zeros(p)
    z=np.zeros((n,p))
    data_copy=data_ori.copy()
    data_copy[:,np.all(np.isnan(data_copy),axis=0)] = 0
    mu_obs=np.nanmean(data_copy, axis=0)
    for j in range(p):
        if np.count_nonzero(~np.isnan(data_ori[:,j]))<5: # impute with random values instead of Loh-Wainright when one dimension has less than 5 values
            z[:,j] = data_filled[:,j]
            obs_prob[j]=1.0
        else:
            for i in range(n):
                if np.isnan(data_ori[i,j]):
                    z[i,j]=0
                else:
                    z[i,j]=data_ori[i,j]-mu_obs[j]
            obs_prob[j]=np.count_nonzero(~np.isnan(data_ori[:,j]))/n

    for i in range(p):
        for j in range(i+1):
            if i==j:
                M[i,j]=1/obs_prob[i]
            else:
                M[i,j]=1/(obs_prob[i]*obs_prob[j])
                M[j,i]=M[i,j]
    fin_cov=stat_tools.cov_nearest(np.multiply(np.matmul(np.transpose(z),z)/n,M))
    
    likelihood=np.zeros(3)
    loss_AIC=np.zeros(3)
    loss_BIC=np.zeros(3)
    loss_eBIC=np.zeros(3)
    edge_count=np.zeros(3)
    c=[0.8,0.85,0.9] # lambda candidates
    for i in range(3):
        glasso_cov, glasso_pre = graphical_lasso(fin_cov,c[i])
        ome_=np.copy(glasso_pre)
        det = np.linalg.det(glasso_pre)
        if det<=0:
            ome_ = stat_tools.cov_nearest(glasso_pre)
            det = np.linalg.det(ome_)
        loss = -np.log(det)+np.trace(np.matmul(glasso_pre,fin_cov))
        likelihood[i] = loss*n/2 # negative log-likelihood
        edge_count[i] = p*(p-1)/2-np.count_nonzero(glasso_pre==0)/2
        loss_AIC[i] = likelihood[i]+edge_count[i]
        loss_BIC[i] = likelihood[i]+edge_count[i]*np.log(n)/2
        loss_eBIC[i] = likelihood[i]+edge_count[i]*np.log(n)/2 + 4*edge_count[i]*np.log(p)*0.5 # \gamma=0.5 as recommended value in eBIC paper

    return np.min(loss_AIC),np.min(loss_BIC),np.min(loss_eBIC)

if __name__ == '__main__':
    spar = 's01' # 's01' for sparsity=0.1 (percentage of missing values), 's025' for sparsity=0.25 and 'complete' for no missing values. 
    dim = 'p30'
    for ii in range(1,21,1):
        for typ in ['_miss_1','_miss_2']: # type of missingness. '' for complete data
            try:
                filename = 'data/synthetic_data/'+str(spar)+'/'+str(spar)+'_'+str(dim)+'_'+str(ii)+typ+'.csv'
                gain_path = 'gain/'+str(spar)+'_'+str(dim)+'_baseline_'+str(ii)+typ
                fig_path = gain_path+'.png'
                if os.path.exists(filename):
                    data_file = pd.read_csv(filename)
                    data = np.array(data_file.iloc[:,:])
                    print(np.shape(data))
                    win=40 # window size
                    l=0
                    ind = list(range(int(win/2),np.size(data,0)-int(win/2)))
                    AIC=np.zeros(np.size(data,0))
                    BIC=np.zeros(np.size(data,0))
                    eBIC=np.zeros(np.size(data,0))
                    loss_AIC=np.full((np.size(data,0),np.size(data,0)),np.nan)
                    loss_BIC=np.full((np.size(data,0),np.size(data,0)),np.nan)
                    loss_eBIC=np.full((np.size(data,0),np.size(data,0)),np.nan)
        
                    m = np.isnan(data) # mask
                    data_filled = np.copy(data)
                    data_filled[m] = np.random.normal(0,1,size=m.sum())
            
        
                    with open(gain_path,'w') as f:
                        f.write('Left,Mid,Right,Loss_AIC,Loss_BIC,Loss_eBIC\n')
                        for l in range(np.size(data,0)-win):
                            m=l+int(win/2)
                            r=l+win
                            f.write(str(l)+','+str(m)+','+str(r)+',')
                            if np.isnan(loss_AIC[l,r]):
                                loss_AIC[l,r],loss_BIC[l,r],loss_eBIC[l,r]=data_imputation(np.array(data[l:r]),np.array(data_filled[l:r]))
                            if np.isnan(loss_AIC[l,m]):
                                loss_AIC[l,m],loss_BIC[l,m],loss_eBIC[l,m]=data_imputation(np.array(data[l:m]),np.array(data_filled[l:m]))
                            if np.isnan(loss_AIC[m,r]):
                                loss_AIC[m,r],loss_BIC[m,r],loss_eBIC[m,r]=data_imputation(np.array(data[m:r]),np.array(data_filled[m:r]))
                            AIC[m-1]=loss_AIC[l,r] - (loss_AIC[l,m]+loss_AIC[m,r])
                            BIC[m-1]=loss_BIC[l,r] - (loss_BIC[l,m]+loss_BIC[m,r])
                            eBIC[m-1]=loss_eBIC[l,r] - (loss_eBIC[l,m]+loss_eBIC[m,r])
                            f.write(str(AIC[m-1])+','+str(BIC[m-1])+','+str(eBIC[m-1])+'\n')
                    f.close()
                else:
                    print(str(filename)+'\t\tFile not exist')
            except Exception as e:
                print('TYPE: '+str(typ))
                print('No. '+str(ii))
                print(str(e))
                pass
    
