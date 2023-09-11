from sklearn.cross_decomposition import CCA
import seaborn as sns
import importlib
import numpy as np 
from CCAtools.plotting.plotcca import CircleBarPlot
from CCAtools.plotting.plotting_utils import returnEdges2Mat,calcEdgeSums
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
from statsmodels.stats.multitest import multipletests
import sys
import json
import matlab

class CCA_class:
    def __init__(self,X,Y,Xlabels,Ylabels,nperms,flip=False,pset=0,mlab_eng=True):
        """Class containig a CCA object"""
        self.Xlabels=Xlabels
        self.Ylabels=Ylabels
        self.flip=flip
        self.nperms=nperms
        self.SM_comps=Y
        self.dist_comps=X
        assert mlab_eng!=False,'pass an instance of a matlab runtime ' 
        eng=mlab_eng
        assert ('PermCCA' in eng.path())==True,'add PermCCA to your matlab path'
        assert ('palm' in eng.path())==True,'add PermCCA to your matlab path'
        X=np.ascontiguousarray(X)
        Y=np.ascontiguousarray(Y)
        mlab2np= lambda arr: np.asarray(arr).squeeze()
        if pset!=0:
            print('running with defined permutation block')
            pset=np.loadtxt(pset)[:,1:].tolist()
            pset=matlab.double(pset)
            print('using permutation block')
            X=matlab.double(X.tolist())
            Y=matlab.double(Y.tolist())
            pfwer,r,A,B,U,V=eng.permcca(X,Y,nperms,[],[],[],0,pset,nargout=6)
            mlab_vars=[pfwer,r,A,B,U,V] 
        else:
            print('no permuation block defined. using defaults of permCCA')
            X=matlab.double(X.tolist())
            Y=matlab.double(Y.tolist())
            pfwer,r,A,B,U,V=eng.permcca(X,Y,nperms,nargout=6)
            mlab_vars=[pfwer,r,A,B,U,V]        
        
        self.pfwer,self.r,self.x_loadings_,self.y_loadings_,\
        self.XCanonVar,self.YCanonVar=[mlab2np(mlab_vars[i]) for i in range(len(mlab_vars)) ]


    
    def BackProjectLoadings(self,cv,pcaXloadings,pcaYloadings):
        """back project the CCA and PCA loadings
        cv is the cannonical variate to back project
        pca loadings are the PCA loadings of the X   and Y CCA inputs respectively"""
        
        if self.flip==True:
            x_edgeWeights=np.dot(pcaXloadings,(-1*self.x_loadings_[:,cv].T))
            self.x_varloadings=pd.DataFrame([dict(zip(self.Xlabels,x_edgeWeights))]).T

            y_edgeWeights=np.dot(pcaYloadings.T,(-1*self.y_loadings_[:,cv]))
            self.y_varloadings=pd.DataFrame([dict(zip(self.Ylabels,y_edgeWeights))]).T
        else:
            x_edgeWeights=np.dot(pcaXloadings,self.x_loadings_[:,cv].T)
            self.x_varloadings=pd.DataFrame([dict(zip(self.Xlabels,x_edgeWeights))]).T
            
            y_edgeWeights=np.dot(pcaYloadings.T,self.y_loadings_[:,cv])
            self.y_varloadings=pd.DataFrame([dict(zip(self.Ylabels,y_edgeWeights))]).T
        return self
    
    def BackProjectWeights(self,cv,SM,distData):
        y_g={}
        for i in SM:
            y_g[i]=np.corrcoef(SM[i],self.YCanonVar[:,cv])[0,1]
        self.y_edgeWeights=pd.DataFrame([y_g]).T
        
        x_g={}
        dist=distData.reset_index().drop(columns='Subject').values.T
        for i,j in enumerate(dist):
            x_g[i]=np.corrcoef(j,self.XCanonVar[:,cv])[0,1]
        self.x_edgeWeights=pd.DataFrame([x_g]).T
        return self

    def plot_canoncorr(self,cv):
        """plot the X and Y canonical variates for a given dimension"""
        plt.figure(figsize=(6,3))
        sns.scatterplot(x=self.XCanonVar[:,cv],y=self.YCanonVar[:,cv],c=self.XCanonVar[:,cv],cmap='coolwarm',s=20)
        sns.regplot(x=self.XCanonVar[:,cv],y=self.YCanonVar[:,cv],scatter=False,color='teal')
        corr_text = "r= {:.3f}".format(self.r[cv])
        plt.annotate(corr_text, xy=(0.1, 0.9), xycoords='axes fraction')
        sns.despine()
        plt.xlabel('Distance Canonical Variate')
        plt.ylabel('Behavior Canonical Variate')
        plt.title(f'Canonical Variates: Dimension {cv +1}')
        plt.tight_layout()
    
    @classmethod
    def from_json(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)

        instance = cls.__new__(cls)
        
        for key, value in data.items():
            if isinstance(value, list):
                value_np = np.array(value)
                setattr(instance, key, value_np)
            elif key in ['Xlabels', 'Ylabels']:
                value_pd = pd.Index(value)
                setattr(instance, key, value_pd)
            else:
                setattr(instance, key, value)

        return instance
