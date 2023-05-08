import nibabel as nib 
from surfdist.load import load_cifti_labels
import numpy as np 
from surfdist.analysis import shortest_path,dist_calc


def get_Paths(edges,label,hemi,surf):
    """return a dictionary of paths connecting two edges"""
    cifti_labels=load_cifti_labels(label,hemi)
    cortex=np.arange(0,32492,1)
    cortex=np.delete(cortex,cifti_labels['???'])
    
    verts=nib.load(surf).darrays[0].data
    
    if hemi=='L':
        prefix='7Networks_LH_'
    elif hemi=='R':
        prefix='7Networks_RH_'
        
    paths={}

    for key in edges.keys():
        val=key.split('--')
        start=prefix+val[0]
        end=prefix+val[1]
        start=cifti_labels[start]
        end=cifti_labels[end]
        d=dist_calc(surf,cortex,start)[end]
        end=end[np.argmin(d)]
        
        d=dist_calc(surf,cortex,end)[start]
        start=start[np.argmin(d)]

        path=shortest_path(surf,cortex,start,end)
        paths[key]=path
    return paths