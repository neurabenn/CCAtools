import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.cm as cm
from .plotting_utils import *
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

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

def CCA_BEHPlot(cca_inst):
    
    sm_ax=cca_inst.y_edgeWeights.sort_values(by=0)
    sm_axNeg=np.abs(sm_ax[sm_ax[0]<0])[::-1]
    sm_axPos=np.abs(sm_ax[sm_ax[0]>0])

    cmap = plt.colormaps['Blues']
    neg=sm_axNeg.values.squeeze()
    norm_neg=np.abs(neg)/np.max(np.abs(neg))
    Negcolors = cmap(norm_neg)

    cmap = plt.colormaps['Reds']
    pos=sm_axPos.values.squeeze()
    norm_pos=np.abs(pos)/np.max(np.abs(pos))
    Poscolors = cmap(norm_pos)

    

    # Create the first subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    joint_max=np.max(np.concatenate([neg,pos]))
    # Plot the first bar plot in the left subplot (ax1)
    y = [i * 0.9 for i in range(len(neg))]
    ax1.barh(y, neg, height=0.8, align="edge", color=Negcolors)

    # Customize the first subplot (ax1) as needed
    # ax1.set_xlabel('X Label')
    # ax1.set_ylabel('Y Label')
    ax1.set_title('Negative Weights')

    ax1.spines["left"].set_capstyle("butt")
    ax1.xaxis.set_tick_params(labelbottom=False, labeltop=False, length=1)
    ticks = y[0::10] + [y[-1]]  # Include highest value in y
    labels = [f"{neg_val:.2f}" for neg_val in neg[0::10]] + [f"{neg[-1]:.2f}"]  # Format labels to two decimal places
    ax1.set_yticks(ticks)

    labels=[str(float(i)*-1) for i in labels]
    ax1.set_yticklabels(labels,fontsize=16)


    ### remove the spines 
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["bottom"].set_visible(False)
    ax1.spines["left"].set_lw(1.5)


    #### add the names of the variables 

    names=[i.replace('_',' ') for i in sm_axNeg.index ]
    PAD = 0.01
    for name, count, y_pos in zip(names, neg, y):
        x = 0
        color = "white"
        path_effects = None
    #     path_effects=[withStroke(linewidth=0.6, foreground="white")]
        if count < 0.055:
            x = count
            color = Negcolors[-2]    
    #         path_effects=[withStroke(linewidth=0.25, foreground="gray")]

        ax1.text(
            x + PAD, y_pos + 0.7/ 2, name, 
            color=color, fontsize=10, fontfamily="Sans",fontweight='regular',va="center",
            path_effects=path_effects
        ) 

    ############# subplot 2 ###################

    ax2.set_title('Positive Weights')
    y = [i * 0.9 for i in range(len(pos))]
    ax2.barh(y, pos, height=0.8, align="edge", color=Poscolors)
    ax2.xaxis.set_tick_params(labelbottom=False, labeltop=False, length=1)

    ticks = y[0::8] + [y[-1]]  # Include highest value in y
    labels = [f"{pos_val:.2f}" for pos_val in pos[0::8]] + [f"{pos[-1]:.2f}"]  # Format labels to two decimal places
    ax2.set_yticks(ticks)

    ax2.set_yticklabels(labels,fontsize=16)

    ax2.tick_params(axis='y', labelright=True,labelleft=False)

    ### remove the spines 
    ax2.spines["right"].set_lw(1.5)
    ax2.spines["top"].set_visible(False)
    ax2.spines["bottom"].set_visible(False)
    ax2.spines["left"].set_visible(False)


    #### add the names of the variables 

    names=[i.replace('_',' ') for i in sm_axPos.index ]
    PAD = 0.01
    for name, count, y_pos in zip(names, pos, y):
        x = 0
        color = "white"
        path_effects = None
    #     path_effects=[withStroke(linewidth=0.6, foreground="white")]
        if count < 0.04:
            x = count
            color = Poscolors[-8]    
    #         path_effects=[withStroke(linewidth=0.25, foreground="gray")]

        ax2.text(
            x + PAD, y_pos + 0.7/ 2, name, 
            color=color, fontsize=10, fontfamily="Sans",fontweight='regular',va="center",
            path_effects=path_effects
        ) 


    plt.tight_layout()
    fig.subplots_adjust(wspace=0.1)

    # Display the figure
    plt.tight_layout()
    plt.suptitle('CCA Axis of Behavior')