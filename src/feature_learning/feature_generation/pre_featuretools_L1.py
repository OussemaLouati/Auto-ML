import featuretools as ft
import pandas as pd


class PreFeaturetoolsL1:
    """
        PreFeaturetoolsL1
        -------------
        Methods:
        --------
        Entityset([name]): Create an Entityset\\
        get_index_candidates([dataframe]): Return index column of a dataframe\\
        detect_relationships([dataframes]): detect relationships between dataframes
        """
    def __init__(self):
        pass

    @classmethod
    def Entityset(cls,name: str =None)-> ft.EntitySet:
        '''Creates and Entityset Stores all actual data for a entityset
        parameters 
         ----------
            • name(str) - Name of the entityset.
        
         returns
         --------
            EntitySet
        '''
        return ft.EntitySet(id =name)
    @classmethod
    def get_index_column(cls,dataframe: pd.DataFrame =None)->str:
        '''Return index column of a dataframe
        parameters 
         ----------
            • dataframe (pd.DataFrame) - our DataFrame.
        
         returns
         --------
            str : column name
        
        '''
        list=[]
        for c in dataframe.columns:
            if len(dataframe[c].unique())==dataframe[c].shape[0]:
                list.append(c)
        return list

    @classmethod                    
    def detect_relationships(cls,dataframes : 'list of DataFrames' =[])-> dict:
        '''Return a dictionary of relationships between our Dataframes
        parameters 
         ----------
            • dataframes (list[pd.DataFrame,pd.DataFrame,pd.DataFrame,...]) - list of dataframes.
        
         returns
         --------
            dict : Dictionary of relationships
        
        '''
        relations=dict()
        i=1
        for df in dataframes[:len(dataframes)-1]:
            for dt in dataframes[i:]:
                #intersection of the name of columns between our two dataframes
                col=[value for value in list(df.columns) if value in list(dt.columns)] 
                if ((len(col)==1)):
                    #the dataframe with less rows will be the parent dataframe
                    parent= df.shape[0] < dt.shape[0]
                    if parent :
                        #index (i-1) of parent should come first 'i-1'
                        relations[col[0]]=[i-1,i]
                    else:
                        relations[col[0]]=[i,i-1]
            i=i+1
        return relations
