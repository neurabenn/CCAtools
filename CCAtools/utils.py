import pandas as pd 
import nibabel as nib 
import numpy as np 
def loadData(filepath):
    """ load the data matrices."""
    data=pd.read_csv(filepath).set_index('Unnamed: 0').T
    return  data    

def save_gifti(data,out):
	"""Save gifti file providing a numpy array and an output file path"""
	gi = nib.gifti.GiftiImage()
	da = nib.gifti.GiftiDataArray(np.float32(data), intent=0)
	gi.add_gifti_data_array(da)
	nib.save(gi,f'{out}.func.gii')
