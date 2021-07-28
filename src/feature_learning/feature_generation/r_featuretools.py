import featuretools as ft
import pandas as pd
from featuretools.primitives import *
import nlp_primitives
from nlp_primitives import (
    DiversityScore,
    LSA,
    MeanCharactersPerWord,
    PartOfSpeechCount,
    PolarityScore, 
    PunctuationCount,
    StopwordCount,
    TitleWordCount,
    UniversalSentenceEncoder,
    UpperCaseCount)


class Featuretools:
    """
        Featuretools
        -------------
        Methods:
        --------
        Run_dfs([entitySet,max_depth,....]): Run Deep feature Synthesis
    
    """
    def __init__(self):
        pass

    @classmethod    
    def Run_dfs(cls,entityset: ft.EntitySet =None, target_entity: str =None, max_depth: int =2,add_NLP: bool =False,add_all_primitives: bool=False)->pd.DataFrame :  
        """Generate new features with deep feature synthesis (DFS)

        parameters :
        ------
        • target_entity (str)                                           - ’Entity id’ of entity on which to make predictions.\\
        • Entityset  (ft.Entityset)                                          - Entityset.\\
        • max depth (int)                                               - Maximum allowed depth of features.
                                                                            Default: 2\\
        •add_NLP (bool)                                                  - Add NLP techniques Default: False\\
        •add_all_primitives (bool)                                       - add all types of primitives Default: False

        Returns
        ----------
        Dataframe : New Dataframe with new generated columns

        """
        trans_primitives=[]
        agg_primitives=["mean","trend","count","mode","median","sum","avg_time_between", "max", "min", "std", "skew","time_since_first","time_since_last",
                        "n_most_common","entropy"]
        #Basic Transformation primitives  
        if add_all_primitives:       
            trans_primitives=["divide_numeric","num_words","time_since","cum_max","num_characters","divide_by_feature","is_weekend","longitude",
                           "add_numeric", "week","and","or","percentile","second","day","diff","time_since_previous","isin","modulo_numeric_scalar",
                           "modulo_by_feature","weekday","haversine","month","latitude","minute","hour"]
        #Transformation primitives
        if add_NLP:
            #NLP primitives
            Nlp_trans = [DiversityScore, LSA, MeanCharactersPerWord, PartOfSpeechCount, PolarityScore, PunctuationCount,
                        StopwordCount, TitleWordCount, UniversalSentenceEncoder,UpperCaseCount]
            trans_primitives=trans_primitives+Nlp_trans
        #aggregation primitives
        if add_all_primitives:  
            feature_matrix, features_defs = ft.dfs(
                                                entityset=entityset,
                                                target_entity=target_entity,
                                                agg_primitives=agg_primitives,
                                                trans_primitives=trans_primitives,
                                                max_depth=max_depth ,
                                                chunk_size=.05                                               )
        else:
            feature_matrix, features_defs = ft.dfs(
                                                entityset=entityset,
                                                target_entity=target_entity,
                                                max_depth=max_depth ,
                                                chunk_size=.05                                               )
        return feature_matrix, features_defs
