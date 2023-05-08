import pandas as pd 
import numpy as np 
import datetime
from sklearn.cross_decomposition import CCA

class hcp_behavior:
    """ """
    def __init__(self,data,subjectList,Larea,Rarea):
        
        self.dataPath=data
        self.subjectList=subjectList
        
        with open(self.subjectList,'r') as f:
            subjects=f.readlines()
        self.subjectList=[np.int64(i.strip('\n')) for i in subjects]
        
        self.dataAll=pd.read_csv(self.dataPath)
        self.dataAll.set_index('Subject',inplace=True)
        self.dataCleanSubj=self.dataAll.loc[self.subjectList]
        
        self.FS=self.dataCleanSubj.iloc[:,190:390]

        self.Larea=pd.read_csv(Larea)
        self.Rarea=pd.read_csv(Rarea)        
        self.Alertness=self.dataCleanSubj.iloc[:,86:95]
        ### move on now 
        self.Cognition=self.dataCleanSubj.iloc[:,114:166]
        self.Emotion=self.dataCleanSubj.iloc[:,166:190]
        #### task performance
        self.TaskEmotion=self.dataCleanSubj.iloc[:,390:395]
        self.TaskGambling=self.dataCleanSubj.iloc[:,395:410]
        self.TaskLanguage=self.dataCleanSubj.iloc[:,410:418]
        self.TaskRelational=self.dataCleanSubj.iloc[:,418:424]
        self.TaskSocial=self.dataCleanSubj.iloc[:,424:445]
        self.TaskWorkingMemory=self.dataCleanSubj.iloc[:,445:499]
        ### back to categories 
        self.Motor=self.dataCleanSubj.iloc[:,499:506]
        
        ### personality has composite scores and answeres for individual cateogry 
        self.PersonalityScores=self.dataCleanSubj.iloc[:,506:511] 
        self.PersonalityRaw=self.dataCleanSubj.iloc[:,511:571] ### will need to encode to do any prediction. 
        encKeys = {'SD':0,'D':1,'N':3,'A':4,'SA':5,np.nan:np.nan}
        for i in self.PersonalityRaw:
            self.PersonalityRaw[i]=[encKeys[j] for j in self.PersonalityRaw[i]]
        
        self.Sensory=self.dataCleanSubj.iloc[:,571:]