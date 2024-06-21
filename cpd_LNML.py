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

def get_G(S,Theta,n,p):
    G=np.zeros((p,p))
    S_inv=np.linalg.inv(S)
    for i in range(p):
        for j in range(p):
            if i != j:
                G[i][j]=(n-p-1)*S_inv[i][j]-n*Theta[i][j]
    return G

def get_h(S,Theta,lmb):
    return np.trace(np.matmul(S,Theta))-np.log(np.linalg.det(Theta))+np.sum(np.multiply(lmb,np.absolute(Theta)))

def get_loss(S_valid,omega,lmbd,valid_size,with_Z):
    # loss is in the LNML code length (with Z)
    ome_=np.copy(omega)
    det = np.linalg.det(omega)
    if det<=0:
        ome_ = stat_tools.cov_nearest(omega)
        det = np.linalg.det(ome_)
    loss = -np.log(det)+np.trace(np.matmul(omega,S_valid))+np.sum(np.multiply(lmbd,np.absolute(omega)))
    if with_Z == True:
        Z=0
        p=omega.shape[0]
        tmp=np.zeros((p,p))
        tmp=np.ones((p,p))*0.5+lmbd
        for i in range(p):
            for j in range(p):
                Z=Z+np.log(tmp[i,j])-np.log(lmbd[i,j])
        return loss*valid_size/2+Z
    else:
        return loss*valid_size/2

def PCLA(sigma,cov_valid,n,n_valid,p,lmbd,T,sig,eta):
    # sigma: emperical covariance
    # cov_valid: emperical covariance for the validation part
    # n: length of (train) data, p:dimensionality
    # n_valid: length of valid data
    # lmbd: initialized penalty matrix
    # T: maximum interations

    # initialize
    eps=0.001
    S=sigma	# emperical covariance
    S_t=np.identity(p)	# S(0)
    result=glasso(emp_cov=S_t,lmb=lmbd,sigma_init=np.eye(p),theta_init=np.eye(p),tol=1e-2,iter_max=50,verbose=True,eps=1e-3)
    Theta_bar=result.theta
    V=np.zeros((p,p))
    Theta_prev=np.zeros((p,p))
    sig_t=sig
    lmbd_t=lmbd
    Delta=np.zeros((p,p))

    lmbd_best=lmbd_t
    sigma_best=S
    theta_best=Theta_bar
    tolar_counter=0
    last_loss=math.inf

    itera=1
    while(itera<T):
        # generate N
        N=np.zeros((p,p))
        for i in range(p):
            for j in range(i+1):
                if i != j:
                    N[i][j]=np.random.normal(0,1,1)
                    N[j][i]=N[i][j]
        # update G
        G_hat=get_G(S_t,Theta_bar,n,p)
        # updata S_tilde
        sig_t=sig_t*np.linalg.det(S_t)
        S_tilde=S_t+sig_t*N+np.power(sig_t,2)*G_hat/2
        # update Theta_tilde
        result=glasso(emp_cov=S_tilde,lmb=lmbd_t,sigma_init=np.eye(p),theta_init=np.eye(p),tol=1e-2,iter_max=50,verbose=True,eps=1e-3)
        Theta_tilde=result.theta

        if np.all(np.linalg.eigvals(S_tilde) > 0):	# if S_tilde is positive definite
            # update h
            h_hat=get_h(S_t,Theta_bar,lmbd_t)
            h_tilde=get_h(S_tilde,Theta_tilde,lmbd_t)
            # update G_tilde
            G_tilde=get_G(S_tilde,lmbd_t,n,p)

            rho=np.linalg.det(S_t)/np.linalg.det(S_tilde)
            beta_q=n*(h_tilde-h_hat)/2+(n-p-1)*np.log(rho)/2
            beta_pi= np.power(np.linalg.norm(rho*N+sig*(np.power(rho,2)*G_hat+G_tilde)/2, 'fro'),2)/2 + np.power(np.linalg.norm(N, 'fro'),2)/2 + p*(p-1)*np.log(rho)/2
            z=np.random.binomial(1,min(1,np.exp(-beta_q-beta_pi)))

            # update S_t
            S_t=z*S_tilde+(1-z)*S_t
            # update Theta_bar
            Theta_bar=z*Theta_tilde+(1-z)*Theta_bar
        # end if

        # update Delta
        result=glasso(emp_cov=S,lmb=lmbd_t,sigma_init=np.eye(p),theta_init=Theta_bar,tol=1e-2,iter_max=50,verbose=True,eps=1e-3)
        Theta_bar_S=result.theta
        Sigma_S=result.sigma
        Delta=np.absolute(Theta_bar_S)-np.absolute(Theta_bar)
        # update V
        for i in range(p):
            for j in range(p):
                V[i][j]=V[i][j]+np.power(Delta[i][j],2)
        # update lambda
        for i in range(p):
            for j in range(p):
                if Delta[i][j] != 0: # to make sure the dinominator is not zero
                    lmbd_t[i][j]=lmbd_t[i][j]-eta*Delta[i][j]/np.sqrt(V[i][j])
                    if lmbd_t[i][j]>1:
                        lmbd_t[i][j]=1
                    if lmbd_t[i][j]<=0.01:
                        lmbd_t[i][j]=0.01
        # update Theta_bar
        result=glasso(emp_cov=S_t,lmb=lmbd_t,sigma_init=np.eye(p),theta_init=np.eye(p),tol=1e-2,iter_max=50,verbose=True,eps=1e-3)
        Theta_bar=result.theta

        lmbd_best=lmbd_t
        sigma_best=Sigma_S
        theta_best=Theta_bar_S
           
        itera=itera+1
    return theta_best,sigma_best,lmbd_best

def data_imputation(data_ori,data_filled):
    # input: original time series data with missing values and data filled by N(0,1)
    # output: time series data after imputation

    with_CV=True
    eps=1.0e-3
    n=np.size(data_ori,0)	# length of time series
    p=np.size(data_ori,1)	# dimensionality
    print("Start data imputation: length "+str(n)+"\t dim "+str(p))

    # data imputation method : Loh-Wainright
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
    
    PCLA_omega_noCV,PCLA_sigma_noCV,PCLA_lmb_noCV=PCLA(fin_cov,fin_cov,n,n,p,np.ones((p,p))*0.5,30,0.574,0.01)  # PCLA(sigma,n,p,lmbd,T,sig,eta)
    loss=get_loss(fin_cov,PCLA_omega_noCV,PCLA_lmb_noCV,n,with_Z=True)
    return loss


if __name__ == '__main__':
    spar = 's01' # 's01' for sparsity=0.1 (percentage of missing values), 's025' for sparsity=0.25 and 'complete' for no missing values. 
    dim = 'p30'
    for ii in range(1,21,1):
        for typ in ['_miss_1','_miss_2']: # type of missingness. '' for complete data
            filename = 'data/synthetic_data/'+str(spar)+'/'+str(spar)+'_'+str(dim)+'_'+str(ii)+typ+'.csv'
            gain_path = 'gain/'+str(spar)+'_'+str(dim)+'_'+str(ii)+typ
            fig_path = gain_path+'.png'
            if os.path.exists(filename):
                data_file = pd.read_csv(filename)
                data = np.array(data_file.iloc[:,:])
                print(np.shape(data))
                win=40 # full-window size
                l=0
                ind = list(range(int(win/2),np.size(data,0)-int(win/2)))
                gain=np.zeros(np.size(data,0))
                loss=np.full((np.size(data,0),np.size(data,0)),np.nan)
        
                m = np.isnan(data) # mask
                data_filled = np.copy(data)
                data_filled[m] = np.random.normal(0,1,size=m.sum()) # fill the positions with missingness with random N(0,1)
            
        
                with open(gain_path,'w') as f:
                    f.write('Left,Mid,Right,Loss,Loss_left,Loss_right,gain\n')
                    for l in range(np.size(data,0)-win):
                        m=l+int(win/2)
                        r=l+win
                        f.write(str(l)+','+str(m)+','+str(r)+',')
                        if np.isnan(loss[l,r]):
                            loss[l,r]=data_imputation(np.array(data[l:r]),np.array(data_filled[l:r]))
                        f.write(str(loss[l,r])+',')
                        if np.isnan(loss[l,m]):
                            loss[l,m]=data_imputation(np.array(data[l:m]),np.array(data_filled[l:m]))
                        f.write(str(loss[l,m])+',')
                        if np.isnan(loss[m,r]):
                            loss[m,r]=data_imputation(np.array(data[m:r]),np.array(data_filled[m:r]))
                        f.write(str(loss[m,r])+',')
                        gain[m-1]=loss[l,r] - (loss[l,m]+loss[m,r])
                        #print("Loss full: "+str(loss[l,r])+"\tLoss left: "+str(loss[l,m])+"\tLoss right: "+str(loss[m,r]))
                        #print("gain["+str(m-1)+"]: "+str(gain[m-1]))
                        f.write(str(gain[m-1])+"\n")
                f.close()
        
                DF=pd.DataFrame()
                DF['gain'] = gain[int(win/2)-1:np.size(data,0)-int(win/2)-1]
                DF.index = ind
                plt.plot(DF['gain'],label='uLNML')
                plt.legend()
                plt.savefig(fig_path)
                plt.clf()

            else:
                print(str(filename)+'\t\tFile not exist')
    
