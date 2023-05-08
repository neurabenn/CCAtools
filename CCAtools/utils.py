import pandas as pd 

def loadData(filepath):
    """ load the data matrices."""
    data=pd.read_csv(filepath).set_index('Unnamed: 0').T
    return  data