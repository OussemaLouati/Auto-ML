import featuretools as ft
from .pre_featuretools_L1 import PreFeaturetoolsL1 as PFL1
import pandas as pd

class PreFeaturetoolsL2:
    """
        PreFeaturetoolsL2
        -------------
        Methods:
        --------
        Add_df_to_entityset([dataframe,entitySet,entity_id]): Add dataframe to an entityset
    
    """
    def __init__(self):
        pass



    @classmethod
    def Add_df_to_entityset(cls,dataframe: pd.DataFrame =None, entitySet: ft.EntitySet =None, entity_id: str =None)-> ft.EntitySet:
        '''Add dataframe to entityset
        parameters 
         ----------
            • dataframe (pd.DataFrame) - dataframe.
            • entityset (ft.entitySet) - entityset
            • entity id (str)          - name of the entityset
        
         returns
         --------
            EntitySet
        '''
        #detect index column
        x=PFL1.get_index_column(dataframe)
        time_ind=None
        if len(x)==0 : 
            index= '...'
            make_index= True 
        elif len(x)==2: 
            if dataframe.dtypes[x[0]]=='<M8[ns]' :
                index= x[1] 
                time_ind= x[0] 
            elif dataframe.dtypes[x[1]]=='<M8[ns]':
                index= x[0] 
                time_ind= x[1] 
            else : 
                index= x[0] 
                time_ind= None 

            make_index= False
        else :
            index= x[0]
            make_index= False


        return entitySet.entity_from_dataframe(entity_id=entity_id, dataframe=dataframe, index = index,make_index=make_index,time_index=time_ind )  
