#!/usr/bin/env python3
import pandas as pd 
import numpy as np 
import datetime
import matplotlib.pyplot as plt
from behavior_class import hcp_behavior
import nibabel as nib 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from scipy.linalg import null_space,svd


import matlab

# def add_mlabPAths(package_list):
#     for i in package_list:
#         eng.addpath(i)

def norm_time(tme):
    tme=datetime.datetime.strptime(tme,'%H:%M:%S')
    minute=tme.minute/60
    return tme.hour+minute

def prep_corr(dists,interest):
#     distVert=pd.read_csv(dists)
    distVert=dists
    distVert.set_index(interest.index.copy(),inplace=True)
    
    valid=interest[~interest.isna()].index
#     print(valid.shape)
#     distVert=distVert.T[valid]
    return interest.T[valid],distVert.T[valid].T

def cube_root(x):
    return x**(1/3)

def normal_eqn_python(X,Y):
    X=np.asarray(X,dtype='float32')
    Y=np.asarray(Y,dtype='float32')
    params=np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(Y)
    resid=Y-np.dot(params.T,X.T).T
    return resid


# varsd=palm_inormal(vars); % Gaussianise
def gauss_SM(data,eng):
    mat_data=matlab.double(data.values.tolist()) ### they actually included the binary acquisition data in the gaussianization
    gaussed=np.asarray(eng.palm_inormal(mat_data))
    return gaussed


def zscore(x,ax=0):
    """z normalize on the first or second axis of the data set"""
    if ax==0:
        print('column wise normalization')
        z=(x-np.nanmean(x,axis=0))/np.nanstd(x,axis=0)
    elif ax==1:
        print('row wise normalization')
        z=(x.T-np.nanmean(x,axis=1))/np.nanstd(x,axis=1)
    
    z[~np.isfinite(z)]=0
    return z

def prep_confounds(confs,eng):
    """ set the confounds up with gaussianization and normalization as done by smith et al 2015."""
    assert ('palm' in eng.path())==True,'add PermCCA to your matlab path'
    mat_data=matlab.double(confs.values.tolist()) ### they actually included the binary acquisition data in the gaussianization
    print('gaussianizing')
    gaussed=np.asarray(eng.palm_inormal(mat_data))
    squared=gaussed[:,1:]**2   
    ready_confs=np.hstack([gaussed,squared])
    ready_confs=zscore(np.hstack([gaussed,squared]),ax=0)

    return ready_confs

def set_confounds(data,Larea,Rarea,mlab_eng):
    """takes in a full data set of all HCP variables and extracts and preprocesses confounds to be regressed"""
    eng=mlab_eng
    confounds=data[['Acquisition','Age_in_Yrs','Height','Weight','BPSystolic','BPDiastolic',
                    'FS_IntraCranial_Vol','FS_BrainSeg_Vol','Larea','Rarea']]
    acq=LabelEncoder().fit_transform(confounds['Acquisition'])
    acq[acq<2]=0
    acq[acq>0]=1  
    df=confounds.copy()
    df['Acquisition']=acq
    confounds=df
    confounds['FS_IntraCranial_Vol']=data['FS_IntraCranial_Vol'].map(cube_root)
    confounds['FS_BrainSeg_Vol']=data['FS_BrainSeg_Vol'].map(cube_root)
    confounds['Larea']=np.sqrt(Larea.values[0])
    confounds['Rarea']=np.sqrt(Rarea.values[0])
    confounds=prep_confounds(confounds,eng)
    return confounds

###### based on anderson winkler's CCA permutation paper 
##### doi: https://doi.org/10.1016/j.neuroimage.2020.117065
def huh_jhun(Z):
    Z=Z.astype(np.float64)
    # Compute the null space of Z'
    nullZ=null_space(Z.T)
    U, D, Vt = svd(nullZ, full_matrices=False, lapack_driver='gesvd')
    Q = U @ np.diag(D)
    # Compute the residual matrix by removing the nuisance variables
    return Q

def preprocess_SM(data,confs,mlab_eng):
    """preprocess the subject measures. Guassianize and remove confounds."""
    eng=mlab_eng
    assert ('palm' in eng.path())==True,'add PermCCA to your matlab path'
    gaussed=gauss_SM(data,eng)
    residuals=normal_eqn_python(confs,gaussed)
    cleaned=zscore(residuals)
    cleaned=pd.DataFrame(cleaned,index=data.index,columns=data.columns)
    return cleaned
def preprocessDists(data,subIDX,confounds):
    
    if type(subIDX[0])==str:
        print('converting strings to ints')
        subIDXint=[int(i) for i in subIDX]
    
    

    NET=data.copy()
    NET=NET.loc[subIDXint]
    dims=NET.shape
    ##### check for vertices with no variance i.e guaranteed masks 
    steady_masks=np.where(np.sum(NET)==0)[0]
    valididx=np.where(np.sum(NET)!=0)[0]
    
    if len(steady_masks)!=0:
        NET=NET.iloc[:,valididx]
        
#     amNET = np.abs(np.nanmean(NET, axis=0))
    NET1 = NET#/amNET
    NET1=NET1-np.mean(NET1,axis=0)
    NET1=NET1/np.nanstd(NET1.values.flatten())
    NET1=normal_eqn_python(confounds,NET1)
    NET1=pd.DataFrame(NET1,columns=NET.columns,index=subIDX)
    
    if len(steady_masks)!=0:
        out=np.zeros(dims)
        out[:,valididx]=NET1.values
        NET1=pd.DataFrame(out,index=NET.index)
    
    return NET1
# def preprocessDists(data,subIDX,confounds):
    
#     if type(subIDX[0])==str:
#         print('converting strings to ints')
#         subIDXint=[int(i) for i in subIDX]
    
#     NET=data.copy()
#     NET=NET.loc[subIDXint]
#     amNET = np.abs(np.nanmean(NET, axis=0))
#     NET1 = NET/amNET
#     NET1=NET1-np.mean(NET1,axis=0)
#     NET1=NET1/np.nanstd(NET1.values.flatten())
#     NET1=normal_eqn_python(confounds,NET1)
#     NET1=pd.DataFrame(NET1,columns=NET.columns,index=subIDX)
#     return NET1

def preprocPCA(data,ncomps):
    data=zscore(data)
    pc=PCA(n_components=ncomps,random_state=42)
    pcTransformed=pc.fit_transform(data)
    explained_var=np.sum(pc.explained_variance_ratio_)
    print(f'PCA  with {ncomps} explains {explained_var:.5f} of variance ')
    return pc,pcTransformed

def loadDataMatrix(filepath,hemi,clean=False):
    """ load the data and optionally clean it. 
    If cleaning pass a tuple containing subject indices in as the first entry 
    and the confounds to be regressed out as the second"""
    
    data=pd.read_csv(filepath).set_index('Unnamed: 0').T
      
    
    if clean:
        subjectIDX=clean[0]
        confounds=clean[1]
        
        print('regressing out the confounds')
        cleanData=preprocessDists(data,subjectIDX,confounds)
        
        return data,cleanData
        
    else:
        print('raw distances loaded')

        return data
