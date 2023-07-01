from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FactorAnalysis
import numpy as np 
import sys
import pandas as pd

def PCA_varimax(ncomps,data):
    pca=PCA(ncomps,random_state=42)
    pca=pca.fit(data)
    PCAcomps=pca.fit_transform(data)
    print(f'{ncomps} components explains')
    
    print(f'{np.sum(pca.explained_variance_ratio_*100):.2f} % of the variance ')
    #f=plt.figure(figsize=(4,2))
    #sns.lineplot(data=pca.explained_variance_ratio_)
    #plt.xlabel('n components')
    #plt.ylabel('var explained')
    
    pcaLoadings=pca.components_.T*np.sqrt(pca.explained_variance_)
    
    fa=FactorAnalysis(rotation='varimax')
    fa.fit(pcaLoadings)
    rotatedLoadings=fa.fit_transform(pcaLoadings)
    return pca,PCAcomps,pcaLoadings,rotatedLoadings

def parseRotatedLoadings(loadings,thr,labels,belowThrehsold=5,percentile=None):
    """ will return the above threshold labels at each loading.
    If no values pass the threshold will return the number of values specified in the belowTrheshold variable
    Returns 3 dictionaries: keys of positive loadings, keys of negative loadings, loadings per component by label
    """
    if len(labels)!=len(loadings):
        print('labels and loadings must match dimensions')
        sys.exit(1)
    
    positive={}
    negative={}
    
    for i in range(loadings.shape[1]):
        if percentile:
            thrPos=np.percentile(loadings[:,i],100-percentile)
            thrNeg=np.percentile(loadings[:,i],percentile)
            posWeigths=np.where(loadings[:,i]>thrPos)[0]
            negWeigths=np.where(loadings[:,i]<(thrNeg))[0]
        else:
            posWeigths=np.where(loadings[:,i]>thr)[0]
            negWeigths=np.where(loadings[:,i]<(thr))[0]
        
        if len(posWeigths)==0:
            posWeigths=np.argsort(loadings[:,i])[::-1][0:belowThrehsold]
        if len(negWeigths)==0:
            negWeigths=np.argsort(loadings[:,i])[0:belowThrehsold]
        
        positive[i]=[labels[p] for p in posWeigths]
        negative[i]=[labels[p] for p in negWeigths]
    return positive,negative,pd.DataFrame.from_dict(dict(zip(labels,loadings)))
