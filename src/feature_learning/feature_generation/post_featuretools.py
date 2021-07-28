import pandas as pd 

class PostFeaturetools:
    ''' 
        PostFeaturetools
        ---------------
        provide :

            \1- Reset index \\
            \2- Delete duplicate columns\\
            \3- Delete columns added by Featuretools
        Methods 
        -------------
        Clean_df([dataframe]) : Clean Dataframe

    '''
    def __init__(self):
        pass

    @classmethod
    def Clean_df(cls,dataframe: pd.DataFrame =None)-> pd.DataFrame:
        """ Clean a dataframe by deleting columns added by featuretools or duplicate columns 
         parameters 
         ----------
            • dataframe (pd.Dataframe) - Dataframe.
        
         returns
         --------
            dataframe : cleaned Dataframe
        """
        #reset index
        dataframe.index=range(dataframe.shape[0])
        #delete columns added by featuretools
        if '...' in dataframe.columns:
            dataframe.drop(columns='...',inplace=True)
        if 'dummy_index' in dataframe.columns:
            dataframe.drop(columns='dummy_index',inplace=True)
        #delete duplicate columns
        d=dict()
        columns=list(dataframe.columns)
        for c in columns:
            d[c]=0
        i=1
        for col in columns[:len(columns)-1]:
            for c in columns[i:]:
                if(dataframe[[col]].equals(dataframe[[c]])):
                    d[c]=1
            i=i+1
        dataframe.drop(columns=[e for e in d if d[e]==1] ,inplace=True) 

        return dataframe
