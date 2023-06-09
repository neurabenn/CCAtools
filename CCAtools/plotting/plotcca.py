import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from .plotting_utils import *

#### add visualizatoin functions here 

def CircleBarPlot(df,key,sort=True):
    """takes in a pandas data frame and uses the key to extract the column to be plotted on the circle
    optionally sort the data from smallest to biggest"""
    if sort:
        df=df.sort_values(by=key,ascending=True)

    ANGLES = np.linspace(0, 2 * np.pi, len(df[key]), endpoint=False)
    VALUES = df[key].values
    LABELS=list(df.T.keys())
    
    # 3 empty bars are added 
    PAD = 3
    ANGLES_N = len(VALUES) + PAD
    ANGLES = np.linspace(0, 2 * np.pi, num=ANGLES_N, endpoint=False)
    WIDTH = (2 * np.pi) / len(ANGLES)
    OFFSET = np.pi / 2
    # The index contains non-empty bards
    IDXS = slice(0, ANGLES_N - PAD)
    
    

    # The layout customization is the same as above
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})

    ax.set_theta_offset(OFFSET)
    ax.set_ylim(np.min(VALUES), np.max(VALUES))
    ax.set_frame_on(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    import matplotlib.cm as cm

    values=df[key].values
    colors = cm.coolwarm((values - np.min(values)) / (np.max(values) - np.min(values)))



    ax.bar(
        ANGLES[IDXS], VALUES, width=WIDTH, color=colors, 
        edgecolor="w", linewidth=2
    )

    add_labels(ANGLES[IDXS], VALUES, LABELS, OFFSET, ax)
    plt.tight_layout()



def writeWBedgeAnnot(outpath,edgeDict,hemi='L',sign='pos'):
    """ will create annotation files for adges and their associated paths
    output path should have a .annot extension"""

    endVerts="</coordList>"
    endLine='</polyLine>'

    if hemi=='L':
        hemi='LEFT'
    else:
        hemi='RIGHT'
    
    with open(outpath, 'w') as file:
        file.write('<?xml version="1.0" encoding="UTF-8"?>'+'\n')
        file.write('<AnnotationFile version="2">'+'\n')
        file.write('\t'+'<MetaData/>'+'\n')
        file.write('\t'+'<group coordinateSpace="SURFACE" groupType="SPACE" tabOrWindowIndex="-1" spacerTabIndex="-1,-1,-1" uniqueKey="1">'+'\n')

        for key in edgeDict:
            file.write('\t'+'\t'+'\t'+new_line(sign)+'\n')
            file.write('\t'+'\t'+'\t'+'\t'+f'<coordList count="{str(len(edgeDict[key]))}">'+'\n')
            for vert in edgeDict[key]:
                file.write('\t'+'\t'+'\t'+'\t'+'\t'+new_vert(vert,hemi)+'\n')
            file.write('\t'+'\t'+endVerts+'\n')
            file.write('\t'+endLine+'\n')
        file.write('\t'+'</group>'+'\n')
        file.write('</AnnotationFile>')