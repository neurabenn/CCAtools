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

class CCA_class:
    def __init__(self,X,Y,Xlabels,Ylabels,ncomps,flip=False,nperms=0,pset=0,mlab_eng=False):
        """Class containig a CCA object"""
        self.Xlabels=Xlabels
        self.Ylabels=Ylabels
        self.flip=flip
        self.nperms=nperms
        eng=mlab_eng
        if nperms!=0:
            if importlib.util.find_spec('matlab')==None:
                print('running with permutations uses the matlab engine\n make sure to also  add permCCA to your path \n https://github.com/andersonwinkler/PermCCA \n Will run CCA without permutations with sklearn ')
                self.ccModel=CCA(n_components=ncomps)
                self.ccModel=self.ccModel.fit(X,Y)
                self.XCanonVar,self.YCanonVar=self.ccModel.transform(X,Y)
                self.x_loadings_=self.ccModel.x_loadings_
                self.y_loadings_=self.ccModel.y_loadings_
            else: 
                assert ('PermCCA' in eng.path())==True,'add PermCCA to your matlab path'
                assert mlab_eng!=False,'pass an instance of a matlab runtime ' 
                import matlab
                X=np.ascontiguousarray(X)
                Y=np.ascontiguousarray(Y)

                mlab2np= lambda arr: np.asarray(arr).squeeze()

                if pset!=0:
                    import matlab
                    print('running with defined permutation block')
                    pset=np.loadtxt(pset)
                    pset=matlab.double(pset)
                    print('using permutation block')
                    pfwer,r,A,B,U,V=eng.permcca(X,Y,nperms,[],[],[],0,pset,nargout=6)
                else:
                    print('no permuation block defined. using defaults of permCCA')
                    X=matlab.double(X.tolist())
                    Y=matlab.double(Y.tolist())
                    pfwer,r,A,B,U,V=eng.permcca(X,Y,nperms,nargout=6)

                mlab_vars=[pfwer,r,A,B,U,V]
                
                self.pfwer,self.r,self.x_loadings_,self.y_loadings_,\
                self.XCanonVar,self.YCanonVar=[mlab2np(mlab_vars[i]) for i in range(len(mlab_vars)) ]


    
    def BackProject(self,cv,pcaXloadings,pcaYloadings):
        """back project the CCA and PCA loadings
        cv is the cannonical variate to back project
        pca loadings are the PCA loadings of the X   and Y CCA inputs respectively"""
        
        if self.flip==True:
            x_edgeCorr=np.dot(pcaXloadings,(-1*self.x_loadings_[:,cv].T))
            self.x_edgeCorr=pd.DataFrame([dict(zip(self.Xlabels,x_edgeCorr))]).T

            y_edgeCorr=np.dot(pcaYloadings.T,(-1*self.y_loadings_[:,cv]))
            self.y_edgeCorr=pd.DataFrame([dict(zip(self.Ylabels,y_edgeCorr))]).T
        else:
            x_edgeCorr=np.dot(pcaXloadings,self.x_loadings_[:,cv].T)
            self.x_edgeCorr=pd.DataFrame([dict(zip(self.Xlabels,x_edgeCorr))]).T
            
            y_edgeCorr=np.dot(pcaYloadings.T,self.y_loadings_[:,cv])
            self.y_edgeCorr=pd.DataFrame([dict(zip(self.Ylabels,y_edgeCorr))]).T
        return self
    
    def plotCanonVar(self,cv):
        """create a scatter plot of the canonnical variates
        cv is the index of the component you wish to plo i.e first would be 0"""
        plt.figure(figsize=(4.6,2))
        r=np.corrcoef(self.XCanonVar[:,cv],self.YCanonVar[:,cv])[0,1]
        ax=sns.scatterplot(x=self.XCanonVar[:,cv],y=self.YCanonVar[:,cv],
                           c=self.XCanonVar[:,cv],cmap='coolwarm')
        ax.annotate(f"r = {r:.3f}", xy=(0.05, 0.85), xycoords='axes fraction', fontsize=12)
        return ax


    
    def plotBehavior(self,flip=False):
    
        plt.figure(figsize=(12,3.5))
        ax=sns.barplot(data=self.y_edgeCorr.sort_values(by=0,ascending=False).T,palette='coolwarm_r')
        ax.tick_params(axis='x', rotation=90)
        """plot behavior on a circle graph. Flip multiplies edges by -1"""
        return ax
        # return CircleBarPlot(self.y_edgeCorr,0)
    
    def splitPop(self,cv,threshold,data,canon='x'):
        
        """Splits the population using a threshold and a cannonical variate
    then tests for differences between the population. make sure you use the correct data. 
    i.e. if your PCA input to the CCA was with residualized data, then use that residualized data"""
        #### select the matrix or behavioral data
        if canon=='x':
            self.raw_data=data.T
            canon=self.XCanonVar[:,cv]
            thrP=np.percentile(canon,threshold)
            thrN=np.percentile(canon,100-threshold)
            SubjPos=np.where(canon>thrP)[0]
            SubjNeg=np.where(canon<thrN)[0]
            ### get subject_id's
        
            self.dataMatPos=self.raw_data.iloc[SubjPos]
            self.dataMatNeg=self.raw_data.iloc[SubjNeg]
        else:
            self.raw_data=data
            canon=self.YCanonVar[:,cv]
            thrP=np.percentile(canon,threshold)
            thrN=np.percentile(canon,100-threshold)
            SubjPos=np.where(canon>thrP)[0]
            SubjNeg=np.where(canon<thrN)[0]
            ### get subject_id's
        
            self.dataBehPos=self.raw_data.iloc[SubjPos]
            self.dataBehNeg=self.raw_data.iloc[SubjNeg]
        
        #### split population into sub population

        return self
        
    def t_TestSplitPopMat(self):
        PvalHmat={}
        PvalLmat={}
        PvalHbeh={}
        PvalLbeh={}

        for key in self.dataMatPos.keys():
            ### alpha = 0.025 as alternative is one sided 
            _,pH=ttest_ind(self.dataMatPos[key],
                  self.dataMatNeg[key],alternative='greater')
            
            _,pL=ttest_ind(self.dataMatPos[key],
                            self.dataMatNeg[key],alternative='less') 
            PvalHmat[key]=pH
            PvalLmat[key]=pL
        self.pHmat=pd.DataFrame([PvalHmat]).squeeze()
        self.pLmat=pd.DataFrame([PvalLmat]).squeeze()
        
        return self
    
    def t_TestSplitPopBeh(self):

        PvalHbeh={}
        PvalLbeh={}

        for key in self.dataBehPos.keys():
            ### alpha = 0.025 as alternative is one sided 
            _,pH=ttest_ind(self.dataBehPos[key],
                  self.dataBehNeg[key],alternative='greater')
            
            _,pL=ttest_ind(self.dataBehPos[key],
                            self.dataBehNeg[key],alternative='less') 
            PvalHbeh[key]=pH
            PvalLbeh[key]=pL
        self.pHbeh=pd.DataFrame([PvalHbeh]).squeeze()
        self.pLbeh=pd.DataFrame([PvalLbeh]).squeeze()
        
        return self

    def SignificantEdges(self,correction):

        """applies multiple comparisons correction using statsmodels and returns edges passing alpha 0.025"""
  
        Hpass,self.Hpcorr,_,_=multipletests(self.pHmat,method=correction,alpha=0.025)
        Lpass,self.Lpcorr,_,_=multipletests(self.pLmat,method=correction,alpha=0.025)
        
        if self.flip==True:
            self.EdgesLess=self.pHmat[Hpass]
            self.EdgesGreater=self.pLmat[Lpass]
        else:
            self.EdgesGreater=self.pHmat[Hpass]
            self.EdgesLess=self.pLmat[Lpass]
        return self
    def SignificantBehaviors(self,correction):
        """applies multiple comparisons correction using statsmodels and returns edges passing alpha 0.025"""
        Hpass,self.behHpcorr,_,_=multipletests(self.pHbeh,method=correction,alpha=0.025)
        Lpass,self.behLpcorr,_,_=multipletests(self.pLbeh,method=correction,alpha=0.025)
        
        if self.flip==True:
            self.BehLess=self.pHbeh[Hpass]
            self.BehGreater=self.pLbeh[Lpass]
        else:
            self.BehGreater=self.pHbeh[Hpass]
            self.BehLess=self.pLbeh[Lpass]
        
        return self
    
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
