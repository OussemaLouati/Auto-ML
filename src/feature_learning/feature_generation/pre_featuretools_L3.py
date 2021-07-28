import featuretools as ft
from .pre_featuretools_L2 import PreFeaturetoolsL2 as PFL2

class PreFeaturetoolsL3:
    """
        PreFeaturetoolsL3
        -------------
        Methods:
        --------
        Add_Entities([dataframes,entitySet]): Add a list of dataframes to an entityset
    
    """
    def __init__(self):
        pass

    @classmethod
    def Add_Entities(cls,dataframes: 'list([pd.DataFrame,pd.DataFrame,...])' =None, entitySet: ft.EntitySet =None, Entities_names : 'list of names' =None)->ft.EntitySet:
        '''Add list of dataframes to entityset
        parameters 
         ----------
            • dataframes (list([pd.DataFrame,pd.DataFrame,...])) - list of dataframe.
            • entityset (ft.entitySet) - entityset
            • Entities names (list[(str,str,str)])          - name of the entityset
        
         returns
         --------
            EntitySet
        '''
        i=0
        #if entities names not specified we initialize a list of [None,None,...], for us to be able to run the for loop next
        Entities_names=Entities_names if Entities_names!=None else [None]*len(dataframes)
        for df,name in zip(dataframes,Entities_names):
            #add each dataframe with its name if not specified we give it 'entity_i' as a name
            entitySet=PFL2.Add_df_to_entityset(dataframe=df,entitySet=entitySet,entity_id='entity_'+str(i) if name==None else name)
            i=i+1
        return entitySet
