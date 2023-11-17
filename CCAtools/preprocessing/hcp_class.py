import numpy as np 
import nibabel as nib 
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import pickle

#### some helper funcs 
def recort(X,fill,dims):
    out=np.zeros(dims)
    out[fill]=X
    return out

def gradientOrientation(grad,hemi,aparc):
    """Determine the orientation of the gradients, and also return whether valid for continued study or not"""
    grad=grad #nib.load(grad).agg_data()
    if hemi=='left':
        labels=nib.load(aparc).agg_data()
#         print('getting gradient orientation from left hemisphere')
    else:
        labels=nib.load(aparc).agg_data()
#         print('getting gradient orientation from right hemisphere')
    calc=np.where(labels==45)[0]
    ctr=np.where(labels==46)[0]
    if np.sum(grad[calc])<0 and np.sum(grad[ctr])<0:
#         print('Canonical Orientation DMN at apex')
        return grad,True
    elif np.sum(grad[calc])<0 and np.sum(grad[ctr])>0:
#         print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    elif np.sum(grad[calc])>0 and np.sum(grad[ctr])<0:
#         print(f'REMOVE {subj} FROM STUDY')
        return grad,False
    else:
#         print('flipping gradient orientation for peak detection')
        return grad *-1,True

def dice_it(A,B):
    
    num=2*(len(np.intersect1d(A,B)))
    den=len(A)+len(B)
    
    if den ==0:
        return np.nan
    else:
        return num/den



##### the class 
class hcp_subj:

    def __init__(self,subj,kernel,pca=None,neighbours=None):
        
        self.subj=subj
        
        clusterPath='/well/margulies/projects/data/hcpGrads'
        anatNatPath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/Native'
        anat32Path=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/T1w/fsaverage_LR32k'
        MNIpath=f'/well/win-hcp/HCP-YA/subjectsAll/{subj}/MNINonLinear/fsaverage_LR32k'
        
        self.info=np.load(f'{clusterPath}/{subj}/{subj}.cifti.info.npy',allow_pickle=True).item()
        
        self.dims=self.info['lnverts']
        self.Lfill=self.info['lIDX']
        self.Rfill=self.info['rIDX']
        self.pca=pca
        self.neighbours=neighbours
        
        self.Lsrf=f'{anat32Path}/{subj}.L.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.LnatSrf=f'{anatNatPath}/{subj}.L.midthickness.native.surf.gii'
        self.Lcoords=nib.load(self.Lsrf).darrays[0].data
        self.Lfaces=nib.load(self.Lsrf).darrays[1].data
        
        self.Linflated=f'{anat32Path}/{subj}.L.inflated_MSMAll.32k_fs_LR.surf.gii'
        
        
        self.Rsrf=f'{anat32Path}/{subj}.R.midthickness_MSMAll.32k_fs_LR.surf.gii'
        self.RnatSrf=f'{anatNatPath}/{subj}.R.midthickness.native.surf.gii'
        self.Rcoords=nib.load(self.Rsrf).darrays[0].data
        self.Rfaces=nib.load(self.Rsrf).darrays[1].data

        

        self.Laparc=f'{MNIpath}/{subj}.L.aparc.a2009s.32k_fs_LR.label.gii'
        self.Lsulc=f'{MNIpath}/{subj}.L.sulc.32k_fs_LR.shape.gii'
        
        
        self.LV1=np.where(nib.load(self.Laparc).darrays[0].data==45)[0]
        self.LS1=np.where(nib.load(self.Laparc).darrays[0].data==46)[0]
        self.LA1=np.where(nib.load(self.Laparc).darrays[0].data==75)[0]
        
        
        self.Rinflated=f'{anat32Path}/{subj}.R.inflated_MSMAll.32k_fs_LR.surf.gii'

        
        self.Raparc=f'{MNIpath}/{subj}.R.aparc.a2009s.32k_fs_LR.label.gii'
        self.Rsulc=f'{MNIpath}/{subj}.R.sulc.32k_fs_LR.shape.gii'
        
        self.RV1=np.where(nib.load(self.Raparc).darrays[0].data==45)[0]
        self.RS1=np.where(nib.load(self.Raparc).darrays[0].data==46)[0]
        self.RA1=np.where(nib.load(self.Raparc).darrays[0].data==75)[0]
        
        
    
#         self.LZverts=get_zoneVerts(LWS)
#         self.RZverts=get_zoneVerts(RWS)
    
#         self.LdistSens=np.load(f'{subj}/{subj}.L.dist32K.npy')
#         self.RdistSens=np.load(f'{subj}/{subj}.R.dist32K.npy')
        
        neighbours=self.neighbours
        
        if self.neighbours==None:
            pass
        else:
            self.Lneighbours=SpatialNeighbours(self.Lcoords,self.Lfaces)
            self.Rneighbours=SpatialNeighbours(self.Rcoords,self.Rfaces)
        
        
        if self.pca is None:
           #print('ussing diffusion maps')

            #### full gradient 
            self.grad=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.diffmaps.0{kernel}mm.npy')
            self.Lgrad=self.grad[0][0:len(self.Lfill)]
            self.Lgrad=recort(self.Lgrad,self.Lfill,self.dims)
            self.Lgrad=gradientOrientation(self.Lgrad,'left',self.Laparc)


            self.Rgrad=self.grad[0][len(self.Lfill):]
            self.Rgrad=recort(self.Rgrad,self.Rfill,self.dims)
            self.Rgrad=gradientOrientation(self.Rgrad,'right',self.Raparc)

            ###### session 1 
            ### subsessions
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses1.diffmap.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            ######## session 2 
 
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.mapalign.ses2.s0{kernel}mm.diffmap.npy')
        
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
            
        else:
#             print('using PCA maps')
            ######### load PCA grads
            self.gradses1=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses1.s0{kernel}mm.npy')
            self.Lgradses1=self.gradses1[0][0:len(self.Lfill)]
            self.Lgradses1=recort(self.Lgradses1,self.Lfill,self.dims)
            self.Lgradses1=gradientOrientation(self.Lgradses1,'left',self.Laparc)
    
        
            self.Rgradses1=self.gradses1[0][len(self.Lfill):]
            self.Rgradses1=recort(self.Rgradses1,self.Rfill,self.dims)
            self.Rgradses1=gradientOrientation(self.Rgradses1,'right',self.Raparc)
        
            self.gradses2=np.load(f'{clusterPath}/{subj}/{subj}.pca.ses2.s0{kernel}mm.npy')   
            self.Lgradses2=self.gradses2[0][0:len(self.Lfill)]
            self.Lgradses2=recort(self.Lgradses2,self.Lfill,self.dims)
            self.Lgradses2=gradientOrientation(self.Lgradses2,'left',self.Laparc)
    
        
            self.Rgradses2=self.gradses2[0][len(self.Lfill):]
            self.Rgradses2=recort(self.Rgradses2,self.Rfill,self.dims)
            self.Rgradses2=gradientOrientation(self.Rgradses2,'right',self.Raparc)
        
        
        
    
    def print_subj(self):
        print(self.subj)
    
    
    def extract_topX(self,Left,Right,pct):
        """extract the top X percent instead of binning"""
        
        
        Left=Left[0]
        Right=Right[0]
        Lout=np.zeros(self.dims)
        Rout=np.zeros(self.dims)
        
        Lpct=np.percentile(Left[self.Lfill],pct)
        
        
        Lthr=np.where(Left[self.Lfill]>Lpct)[0]
        Linter=np.zeros(len(self.Lfill))
        Linter[Lthr]=1
        L=recort(Linter,self.Lfill,self.dims)
        L=np.where(L!=0)[0]
        
        #### do right 
        
                
        Rpct=np.percentile(Right[self.Rfill],pct)
        
        
        Rthr=np.where(Right[self.Rfill]>Rpct)[0]
        Rinter=np.zeros(len(self.Rfill))
        Rinter[Rthr]=1
        R=recort(Rinter,self.Rfill,self.dims)
        R=np.where(R!=0)[0]
        

        return L,R
 
  
    
    def dice_Ses12(self,pct):
        S1=self.extract_topX(self.Lgradses1,self.Rgradses1,pct)
        S2=self.extract_topX(self.Lgradses2,self.Rgradses2,pct)
      
        diceL=dice_it(S1[0],S2[0])
        diceR=dice_it(S1[1],S2[1])
        
        
        return np.asarray([diceL,diceR])
