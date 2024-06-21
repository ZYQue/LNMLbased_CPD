import numpy as np
import os.path
import pandas as pd
from scipy import linalg
import matplotlib.pyplot as plt
from sklearn import metrics

def ben_far(gain,cps,beta,T):
    ben=0
    far=0
    a=np.zeros_like(gain)
    a[gain>beta]=1
    ben_arr=np.zeros_like(gain)
    for i in cps:
        for j in range(i-T,i+T-1):
            ben_arr[j]=1-np.abs(j+1-i)/T
    for i in np.where(a==1):
        ben=ben_arr[i].sum()
        far=np.count_nonzero(ben_arr[i]==0) 
    return ben,far


def normalize(s):
    return s/np.max(s)

if __name__ == '__main__':
    auc_likeli=np.zeros(20)
    auc_AIC=np.zeros(20)
    auc_BIC=np.zeros(20)
    auc_eBIC=np.zeros(20)
    ctr=0
    typ='_miss_1' # or '_miss_2' or '_complete'
    for ii in range(1,21,1):
        cps=[40,120]	# positions of change points (half-window size = 20)
        gain_path = 'gain_01_p30_baseline_'+str(ii)+typ
        if(os.path.isfile(gain_path)):      # gains file exists
            log = pd.read_csv(gain_path)
            AIC = log.loc[:,'Loss_AIC']
            BIC = log.loc[:,'Loss_BIC']
            eBIC = log.loc[:,'Loss_eBIC']
            
            T=5
            bens_AIC=np.zeros(100)
            fars_AIC=np.zeros(100)
            bens_BIC=np.zeros(100)
            fars_BIC=np.zeros(100)
            bens_eBIC=np.zeros(100)
            fars_eBIC=np.zeros(100)
            for i in range(100):
                beta_AIC=AIC.min()+i*(AIC.max()-AIC.min())/100
                bens_AIC[i],fars_AIC[i]=ben_far(AIC,cps,beta_AIC,T)
                beta_BIC=BIC.min()+i*(BIC.max()-BIC.min())/100
                bens_BIC[i],fars_BIC[i]=ben_far(BIC,cps,beta_BIC,T)
                beta_eBIC=eBIC.min()+i*(eBIC.max()-eBIC.min())/100
                bens_eBIC[i],fars_eBIC[i]=ben_far(eBIC,cps,beta_eBIC,T)
            bens_AIC = normalize(bens_AIC)
            fars_AIC = normalize(fars_AIC)
            bens_BIC = normalize(bens_BIC)
            fars_BIC = normalize(fars_BIC)
            bens_eBIC = normalize(bens_eBIC)
            fars_eBIC = normalize(fars_eBIC)
            auc_AIC[ctr] = metrics.auc(fars_AIC,bens_AIC)
            auc_BIC[ctr] = metrics.auc(fars_BIC,bens_BIC)
            auc_eBIC[ctr] = metrics.auc(fars_eBIC,bens_eBIC)
            ctr=ctr+1
        else:
            print("No gains file!")
    print("AIC:")
    print(auc_AIC)
    print("mean:")
    print(np.mean(auc_AIC))
    print("deviation:")
    print(np.std(auc_AIC))
    print("BIC:")
    print(auc_BIC)
    print("mean:")
    print(np.mean(auc_BIC))
    print("deviation:")
    print(np.std(auc_BIC))
    print("eBIC:")
    print(auc_eBIC)
    print("mean:")
    print(np.mean(auc_eBIC))
    print("deviation:")
    print(np.std(auc_eBIC))
