from sklearn.cross_decomposition import CCA
import seaborn as sns
import importlib
import numpy as np 
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
from statsmodels.stats.multitest import multipletests
import sys

def splitPopulation(cca_inst,cv):
    SubjPos=np.where((cca_inst.XCanonVar[:,cv] >= 0) & (cca_inst.YCanonVar[:,cv] >= 0))[0]
    SubjNeg=np.where((cca_inst.XCanonVar[:,cv] <= 0) & (cca_inst.YCanonVar[:,cv] <= 0))[0]
    return  SubjPos,SubjNeg




def plot_split(cca_inst,cv,ex_data):
        """plot the X and Y split for t-testing
        provide a cca class instance, the canonical variate you wish to investigate, 
        and either the behavior or distance dataframes input into the """
        df=pd.DataFrame([cca_inst.XCanonVar[:,cv],cca_inst.YCanonVar[:,cv]]).T
        df.set_index(ex_data.index,inplace=True)
        df.rename(columns={0:'X',1:'Y'},inplace=True)
        df['hue']=0
        
        pos_subj,neg_subj=splitPopulation(cca_inst,cv)
        pos_subj=ex_data.index[pos_subj]
        neg_subj=ex_data.index[neg_subj]
        df.loc[pos_subj,'hue']=1
        df.loc[neg_subj,'hue']=-1

        ax=plt.figure(figsize=(6,3))
        sns.scatterplot(x='X',y='Y',hue='hue',data=df,s=20,palette='coolwarm')
        # sns.regplot(x=df['X'],y=df['Y'],data=df,scatter=False,color='teal')
        sns.despine()
        plt.xlabel('Distance Canonical Variate')
        plt.ylabel('Behavior Canonical Variate')
        plt.title(f'Population Split: Dimension {cv +1}')
        plt.tight_layout()