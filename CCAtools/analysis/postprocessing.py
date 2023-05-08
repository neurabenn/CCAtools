import numpy as np 
import pandas as pd
from surfdist.load import load_cifti_labels

####### posthoc get things back into a matrix so you can do network calcs. 
def returnEdges2Mat(data,arraySize):
    """Take a set of edges and palce it back into a full connectivity matrix."""
    out=np.zeros((arraySize,arraySize))
    indices=np.triu_indices((arraySize),k=1)
    out[indices]=data
    out=out+out.T   
    return out

def calcEdgeSums(mat,labelFile,hemi,rm=None):
    labels=load_cifti_labels(labelFile,hemi)
    if rm:
        del labels[rm]
    edgeSums=pd.DataFrame([np.sum(mat,axis=0)],columns=labels.keys())
    
    surf_data=np.zeros(32492)
    for key in labels:
        surf_data[labels[key]]=edgeSums[key]
    
    
    return edgeSums,surf_data

