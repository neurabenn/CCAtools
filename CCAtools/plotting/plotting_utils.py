import numpy as np 
from surfdist.load import load_cifti_labels 
import pandas as pd

def get_label_rotation(angle, offset):
    # Rotation must be specified in degrees :(
    rotation = np.rad2deg(angle + offset)
    if angle <= np.pi:
        alignment = "right"
        rotation = rotation + 180
    else: 
        alignment = "left"
    return rotation, alignment
def add_labels(angles, values, labels, offset, ax):
    
    # This is the space between the end of the bar and the label
#     padding = 0.5
    
    # Iterate over angles, values, and labels, to add all of them.
    for angle, value, label, in zip(angles, values, labels):
        angle = angle
       
        # Obtain text rotation and alignment
        rotation, alignment = get_label_rotation(angle, offset)
        if value<0:
            padding=np.abs(value)+0.01
        else:
            padding=0.01
        

        # And finally add the text
        ax.text(
            x=angle,
            y=value + padding, 
            s=label, 
            ha=alignment, 
            va="center", 
            rotation=rotation, 
            rotation_mode="anchor"
        ) 
def new_vert(vert,hemi):
    """input a vertex and hemisphere to generate the corresponding line of XML for a wb_annot file """
    ### hemi is passed as LEFT or RIGHT
    out=f'<coord x="0.000000" y="0.000000" z="0.000000" structure="CORTEX_{hemi}" numberOfNodes="32492" nodeIndex="{vert}" nodeOffset="1.000000" nodeOffsetVectorType="TANGENT"/>'
    return out


def returnEdges2Mat(data,arraySize):
    """Take a set of edges and palce it back into a full connectivity matrix."""
    out=np.zeros((arraySize,arraySize))
    indices=np.triu_indices((200),k=1)
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

def new_line(color):
    out=f'<polyLine coordinateSpace="SURFACE" backgroundCaretColor="NONE" backgroundCustomRGBA="0.000000;0.000000;0.000000;1.000000" foregroundCaretColor="CUSTOM" foregroundCustomRGBA="{color}" foregroundLineWidth="10" foregroundLineWidthPercentage="1" tabIndex="-1" windowIndex="-1" spacerTabIndex="-1,-1,-1" uniqueKey="2">'
    return out