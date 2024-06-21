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
    auc_lnml=np.zeros(20)
    auc_glasso=np.zeros(20)
    ctr=0
    typ='_miss_1'
    for ii in range(1,21,1):
        cps=[40,120]	# positions of change points (half-window size=20)
        gain_path = 'gain_01_p30_'+str(ii)+typ
        if(os.path.isfile(gain_path)):      # gains file exists
            log = pd.read_csv(gain_path)
            lnml = log.loc[:,'gain']
            
            T=5
            bens_lnml=np.zeros(100)
            fars_lnml=np.zeros(100)
            for i in range(100):
                beta_lnml=lnml.min()+i*(lnml.max()-lnml.min())/100
                bens_lnml[i],fars_lnml[i]=ben_far(lnml,cps,beta_lnml,T)
            bens_lnml = normalize(bens_lnml)
            fars_lnml = normalize(fars_lnml)
            auc_lnml[ctr] = metrics.auc(fars_lnml,bens_lnml)
            ctr=ctr+1
        else:
            print("No gains file!")
    print("LNML:")
    print(auc_lnml)
    print("mean:")
    print(np.mean(auc_lnml))
    print("deviation:")
    print(np.std(auc_lnml))
