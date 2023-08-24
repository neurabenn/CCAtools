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
def zscore(x):
    z=(x-np.mean(x,axis=0))/np.std(x,axis=0)
    return z

def set_confounds(data,Larea,Rarea):
    
    confounds=data[['Acquisition','Gender','Age']].T
    for i in ['Acquisition','Gender','Age']:
        confounds.loc[i]=LabelEncoder().fit_transform(confounds.T[i])
    confounds=confounds.T
    confounds['FS_IntraCranial_Vol']=data['FS_IntraCranial_Vol'].map(cube_root)
    confounds['FS_BrainSeg_Vol']=data['FS_BrainSeg_Vol'].map(cube_root)
    confounds['Larea']=Larea.values[0]
    confounds['Rarea']=Rarea.values[0]
    #### gaussianize the confounds 
    confounds.drop(columns=i,inplace=True)

    qt = QuantileTransformer(n_quantiles=50, random_state=42,output_distribution='normal')
    gaussianized=qt.fit_transform(confounds.iloc[:,3:])
    
    #### add the squares 
    squareTerms=confounds.iloc[:,3:]**2
    
    
    z_scored=zscore(np.asarray(np.hstack([gaussianized,squareTerms]),dtype=np.float32))
    
    
    return np.hstack([confounds.T.iloc[0:3].T.values,z_scored])

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

def preprocess_SM(data,columns,Larea,Rarea,outl=False,method='raw'):
    """preprocess the subject measures removing their confounds. 
    method 'raw' uses the confounds as are and 'hj' uses the huh_jhun method"""
    nas=set(np.where(data[columns].isna()==True)[0])
    IDX=set(range(len(data)))
    # print(IDX)
    if type(outl)==bool:
        print('no outliers')
        valid=list(IDX-nas)
    else:
        # print('yes outliers')
        valid=list(IDX-nas)
        print(f' there are {len(valid)} indices')
        cln = [ele for idx, ele in enumerate(valid) if idx not in outl]
        valid=cln
        print(f'there are {len(valid)} indices')

    print(f'there are {len(valid)} valid indices ')
    confounds=set_confounds(data,Larea,Rarea)
    confounds=confounds[valid]

    if method=='raw':
        confounds=confounds #### they are already zscored
    elif method=='hj':
        confounds=zscore(huh_jhun(confounds))

    Y=np.asarray(data.iloc[valid][columns].values)
    
    qt = QuantileTransformer(n_quantiles=50, random_state=42,output_distribution='normal')
    Y=zscore(qt.fit_transform(Y))
    
    LinMdl=LinearRegression().fit(confounds,Y)
    residuals=Y-LinMdl.predict(confounds)

    return residuals,valid,confounds


def preprocessDists(data,subjIDX,confounds):
    
    X=zscore(confounds)
    Y=np.asarray(data.T.iloc[subjIDX])
    
    # qt = QuantileTransformer(n_quantiles=50, random_state=42,output_distribution='normal')
    Y=zscore(Y)
    
    LinMdl=LinearRegression().fit(X,Y)
    residuals=Y-LinMdl.predict(X)

    return residuals

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
