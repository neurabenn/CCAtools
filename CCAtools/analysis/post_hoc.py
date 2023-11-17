from sklearn.cross_decomposition import CCA
import seaborn as sns
import importlib
import numpy as np 
import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt 
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import ttest_ind
import sys

def splitPopulation(cca_inst,cv):
    SubjPos=np.where((cca_inst.XCanonVar[:,cv] >= 0) & (cca_inst.YCanonVar[:,cv] >= 0))[0]
    SubjNeg=np.where((cca_inst.XCanonVar[:,cv] <= 0) & (cca_inst.YCanonVar[:,cv] <= 0))[0]
    return  SubjPos,SubjNeg

def ttest_cca(cca_inst,cv,data):
    """perform a t-test on a data set using a given CCA dimension to split the population"""
    ###split the population 
    pos,neg=splitPopulation(cca_inst,cv)
    ### get subject ids and extract the split data
    pos_subj=data.index[pos]
    neg_subj=data.index[neg]
    
    pos_data=data.loc[pos_subj]
    neg_data=data.loc[neg_subj]
    
    pl={}
    pl_t={}
    for i in data.keys():
        t,p=ttest_ind(pos_data[i],neg_data[i],equal_var=False,alternative='less')
        pl[i]=p
        pl_t[i]=t
    pl=pd.DataFrame([pl])
    pl_t=pd.DataFrame([pl_t])

    pg={}
    pg_t={}
    for i in data.keys():
        t,p=ttest_ind(pos_data[i],neg_data[i],equal_var=False,alternative='greater')
        pg[i]=p
        pg_t[i]=t
    pg=pd.DataFrame([pg])
    pg_t=pd.DataFrame([pg_t])
    
    return pl,pg,pl_t,pg_t
    
def significanceTesting(cca_inst,cv,distData,behData,side='beh',plot='off',correction='bonferroni',alpha=0.025):
    ### back project to weights of CV 
    cca_inst.BackProjectWeights(cv,behData,distData)

    if side=='beh':
         testdata=behData
    else:
         testdata=distData
    ### t-test values 
    pl,pg,pl_t,pg_t=ttest_cca(cca_inst,cv,testdata)
    ### multiple comparisons correction 
    pl_pass,plcorr,_,_=multipletests(pl.values.squeeze(),alpha=alpha,method=correction)
    pg_pass,pgcorr,_,_=multipletests(pg.values.squeeze(),alpha=alpha,method=correction)
    t_set=pd.concat([pl_t.iloc[:,pl_pass].T,pg_t.iloc[:,pg_pass].T])
    t_set.rename(columns={0:'t-score'},inplace=True)
    
#     ### get the edge labels back for significant values 
    if side=='beh':
         t_set['CCA_weights']=cca_inst.y_edgeWeights.loc[t_set.index]
    else:
        idx_weights=t_set.index.map(int)
        weights=cca_inst.x_edgeWeights.loc[idx_weights].values
        t_set['CCA_weights']=weights
    ### sort by CCA weights 
    t_set.sort_values(by='CCA_weights',inplace=True)
    if plot !='off':
        t_ready=t_set.reset_index().sort_values(by='CCA_weights').reset_index().set_index('index')
        scl=MinMaxScaler(feature_range=(-1,1))
        t2plot=pd.DataFrame(scl.fit_transform(t_ready[['t-score','CCA_weights']]),\
                            index=t_ready.index,columns=['T-values','CCA Weights'])
        ### plotting is meant for behavioral data only
        fig, ax1 = plt.subplots(figsize=(6,8))
        sns.heatmap(t2plot, cmap='coolwarm', ax=ax1)
        y_label_font_size = 12  # For y-axis label
        ax1.set_ylabel('Behaviors')
        ax1.text(1.3, 0.5, 'Normalized (-1,1)', transform=ax1.transAxes, verticalalignment='center', rotation=270)
        plt.title('T-value and CCA Weights')
        fig.tight_layout()
    return t_set

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