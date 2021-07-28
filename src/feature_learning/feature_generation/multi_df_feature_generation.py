from .r_featuretools import Featuretools  
import featuretools as ft
from .pre_featuretools_L1 import PreFeaturetoolsL1 as PFL1
from .pre_featuretools_L2 import PreFeaturetoolsL2 as PFL2
from .pre_featuretools_L3 import PreFeaturetoolsL3 as PFL3
from .pre_featuretools_L4 import PreFeaturetoolsL4 as PFL4
from .post_featuretools import PostFeaturetools
import pandas as pd
    


class MDFGeneration:
    """
        MDFGeneration:
        -------------
        Calculates a feature matrix and features given a dictionary of entities
        and a list of relationships. 

        Methods:
        ------------
        generate([dataframe,relations,...]) : generate new features using deep feature synthesis
    """
    def __init__(self):
        pass

    @classmethod
    def generate(cls,dataframes: 'list[(pd.Dataframe, pd.Dataframe, pd.Dataframe)]'=[],
                     target_entity: str =None,
                     relations: 'dict[str -> list[int,int]]' =None,
                     Entities_names: 'list[(str, str, str, str)]'=None,
                     Entityset_name: str =None,
                     max_depth: int =2,
                     add_NLP: bool=False,
                     add_all_primitives: bool=False)->pd.DataFrame:
        """
        Generate new features with deep feature synthesis (DFS)
        parameters :
        ------
        • dataframes (list[(pd.Dataframe, pd.Dataframe, pd.Dataframe)]) - The list of our Dataframes.\\
        • target_entity (str)                                           - ’Entity id’ of entity on which to make predictions. If None
                                                                            the target entity will be detected automatically.\\
        • Entities_name (list[(str, str, str, str)])                    - The names of our entities, the names will
                                                                            be affected to each dataframe in the same order as if dataframes.\\
        • Entityset_name (str)                                          - Entityset.
                                                                            Default: Entityset with id=’dummy name’\\
        • max depth (int)                                               - Maximum allowed depth of features.
                                                                            Default: 2\\
        • relations (dict[str -> list[int,int]])                        - List of relationships between entities. List
                                                                            items are a list with the format (index of parent entity in the dataframes list,index
                                                                            of child entity in the dataframes list). If None the functions will try to detect the
                                                                            relationships between our dataframes but it’s recommended to pass them to the
                                                                            function as a parameters.\\
        • target (str)                                                  - The name of the target column\\
        •add_NLP (bool)                                                  - Add NLP techniques Default: False\\
        •add_all_primitives (bool)                                       - add all types of primitives Default: False
        Returns
        ----------
        Dataframe : New Dataframe with new generated columns

        """
        #Create an entitySet
        entityset=PFL1.Entityset(name= 'dummy_name' if  Entityset_name== None else Entityset_name)

        #if no relations are specified we try to detect the relationships automatically
        if relations == None :
            relations=PFL1.detect_relationships(dataframes=dataframes)
        #add entities to our entityset
        entityset=PFL3.Add_Entities(dataframes=dataframes,entitySet=entityset,Entities_names=Entities_names)

        #if no target entity is specified , we detect the target entity ( the one with bigger number of rows)
        if target_entity==None:
            target_entity=PFL4.Detect_target_entity(entityset=entityset)
        #add relationships between entities to our entityset
        entityset=PFL4.Add_relationships(relations=relations,entityset=entityset,Entities_names=Entities_names)

        #run dfs
        df, feats = Featuretools.Run_dfs(entityset=entityset,target_entity=target_entity,max_depth=max_depth,add_NLP=add_NLP,add_all_primitives=add_all_primitives)

        df=PostFeaturetools.Clean_df(dataframe=df)
        return df,feats,entityset
