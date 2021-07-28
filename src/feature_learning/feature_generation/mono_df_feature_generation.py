from .r_featuretools import Featuretools  
import featuretools as ft
from .pre_featuretools_L1 import PreFeaturetoolsL1 as PFL1
from .pre_featuretools_L2 import PreFeaturetoolsL2 as PFL2
from .pre_featuretools_L3 import PreFeaturetoolsL3 as PFL3
from .pre_featuretools_L4 import PreFeaturetoolsL4 as PFL4
from .post_featuretools import PostFeaturetools
from featuretools.autonormalize import autonormalize as an
import pandas as pd
from time import sleep
from tqdm import tqdm 
from feature_learning.feature_selection.selection import Selection
import random

class SDFGeneration:
    """
        SDFGeneration:
        -------------
        Calculates a feature matrix and features after normalizing a dataset

        Methods:
        ------------
        generate([df,levels,...]) : generate new features using deep feature synthesis
    """
    def __init__(self):
        pass

    @classmethod
    def generate(cls,df: pd.DataFrame =None,
                    Entityset_name: str =None,
                    max_depth: int =2, 
                    levels: int =None,
                    Target: str =None,
                    skip_norm: bool =False,
                    add_NLP: bool=False,
                    add_all_primitives: bool=False)->pd.DataFrame:
        """
        Generate new features with deep feature synthesis (DFS)
        parameters :
        ------
        • df (pd.Dataframe) - our Dataframe.\\
        • target_entity (str)                                           - ’Entity id’ of entity on which to make predictions. If None
                                                                            the target entity will be detected automatically.\\
        • Entityset_name (str)                                          - Entityset.
                                                                            Default: Entityset with id=’dummy name’\\
        • max depth (int)                                               - Maximum allowed depth of features.
                                                                            Default: 2\\
        • target (str)                                                  - The name of the target column
        • skip norm (bool,Optional)                                     - If True we skip the Normaliztion process of our dataset,
                                                                            and This is used only if our len(dataframes)==1.
                                                                            Default: False\\
        •add_NLP (bool)                                                  - Add NLP techniques Default: False\\
        •add_all_primitives (bool)                                       - add all types of primitives Default: False                                                                       
        Returns
        ----------
        Dataframe : New Dataframe with new generated columns

        """
        #skip normalization
        if not skip_norm :
            print(' -- Normalizing Dataframe -- ')
            entityset= an.auto_entityset(df, accuracy=1, name= 'dummy_name' if  Entityset_name== None else Entityset_name)
        #perform normalization    
        else:
            print(' \n-- Skipping Normalization --\n ')    
            entityset=PFL1.Entityset(name= 'dummy_name' if  Entityset_name== None else Entityset_name)
        #if number of entities > 1 that means that the normalizationhave succeeded 
        if len(entityset.entities)>1:  
            #we detect target entity 
            target_entity=PFL4.Detect_target_entity(entityset=entityset)
            #Run DFS
            matrix, feats = Featuretools.Run_dfs(entityset=entityset,target_entity=target_entity,max_depth=max_depth,add_NLP=add_NLP,add_all_primitives=add_all_primitives)
            matrix=PostFeaturetools.Clean_df(dataframe=matrix)
            return matrix, feats
        #of number of entities=1 normalization didn't success , note that we used <= for the the case when 
        # we skip normalization, our entityset will be empty   
        elif len(entityset.entities)<=1 : 
            #initialize a progress bar
            #get columns of the dataframe
            columns=list(df.columns)
            #remove target
            columns.remove(Target)
            #get list of datetime columns in our dataset
            datetime_columns=[]
            for c in df.columns :
                if df.dtypes[c]=='<M8[ns]':
                    datetime_columns.append(c)
            for c in datetime_columns : 
                columns.remove(c)
            print('\ndatetime columns : {}'.format(datetime_columns))
            print(columns)
             #we create a new dataframe with no datetime columns and no target columns       
            dataframe=df.drop(columns=datetime_columns+[Target])
            #if number of levels not specified , we will initialize it with number of columns in our dataframe
            if levels==None : 
                print (" \n-- WARNING : N° of levels not specified you might get a Huge number of features --\n ")
                levels=len(columns)  
            #we return alist of random names of columns , list of 'levels' elements
            columns=random.sample(columns, levels)
            tq = tqdm(columns,leave=True)
            count=0
            for c in tq:
                tq.set_description(c)
                tq.refresh() 
                sleep(1.0)
                #at the last iteration we the datetime columns we just dropped
                #the reason for that is simple, at each iteration the same operation will be applied
                #on these types of columns which will at last generate a lot of duplicate columns, for that reason we just add
                #them at the last iteration
                if count == levels-1: 
                    dataframe=pd.concat([dataframe,df[datetime_columns]],axis=1)   
                #if c is dropped ( like datetime column for example) we added before running DFS
                if c not in dataframe.columns:
                    dataframe=pd.concat([dataframe,df[[c]]],axis=1)
                #we initialize an entityset
                entityset=PFL1.Entityset(name= 'dummy_name' if  Entityset_name== None else Entityset_name)
                #add an index column ( this is added to force featuretools to name our entity 'dummy_index' instead of a very long name which is the concatenation 
                # of all the columns of the dataframe which is unreasonable)
                dataframe['dummy_index']=dataframe.index
                col_name=c+' ID'
                dataframe[col_name]=dataframe[c]
                col=dataframe[c].unique()
                #create new dataframe from the unique number of c which will give us a trivial relationship with this dataframe and our old dataframe
                new_df = pd.DataFrame(data=col, columns=[col_name])
                new_df[c]=new_df[col_name]
                dataframe[col_name]=dataframe[col_name].astype(new_df[col_name].dtypes)
                dataframe.drop(columns=[c],inplace=True)
                #list of the two dataframes
                dataframes=[dataframe,new_df]
                #detect relationships
                relations=PFL1.detect_relationships(dataframes=dataframes)
                #add entities
                entityset=PFL3.Add_Entities(dataframes=dataframes,entitySet=entityset,Entities_names=['dummy_index',col_name])
                #add relationships
                entityset=PFL4.Add_relationships(relations=relations,entityset=entityset,Entities_names=['dummy_index',col_name])
                #run DFS and generate features
                matrix, feats = Featuretools.Run_dfs(entityset=entityset,target_entity='dummy_index',max_depth=max_depth,add_NLP=add_NLP,add_all_primitives=add_all_primitives)
                #clean dataframe for the next iteration
                matrix=PostFeaturetools.Clean_df(dataframe=matrix)
                dataframe=matrix.copy()
                #every 2 iteration we perform a feature selection to reduce the number of features or else after 2 iteration wa can have an 
                #overwhelming number of iteration which is make it impossible to run the loop any further
                if count%2!=0 and count!=levels-1:
                    #we target column just to run feature selection then drop
                    #we just drop features created but highly correlated with old features and keep the rest
                                # dataframe=pd.concat([dataframe,df[[Target]]],axis=1)
                    selector = Selection(skip_target=True)            
                    dataframe=selector.fit_transform(dataframe,df[[Target]])
                count=count+1
        #if levels == 1 we don't want to perform any feature reduction or else our function would be of no use 
        if levels!=1 and levels!=None:
            selector = Selection(skip_target=True)            
            dataframe=selector.fit_transform(dataframe,df[[Target]])
        return dataframe, feats
        




