import pandas as pd
import numpy as np


class Utils:
    def __init__(self):
        pass 

    @classmethod
    def split_X_Y(cls,Dataframe=None,Target=None):
        df_x=Dataframe.drop(columns=[Target])
        df_y=Dataframe[Target]
        return df_x,df_y
    
    
    @classmethod
    def Array_to_dataframe(cls,Array=None,df_y=None):
        df=pd.DataFrame(data=Array, columns=['Feature_'+str(i) for i in range(Array.shape[1])]) 
        df=pd.concat([df,df_y],axis=1)
        return df
