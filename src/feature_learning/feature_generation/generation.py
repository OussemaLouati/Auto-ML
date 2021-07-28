from mono_df_feature_generation import SDFGeneration
from multi_df_feature_generation import MDFGeneration
import pandas as pd 


class FGeneration:
    """ 
        FGeneration
        ----------
        Calculates a feature matrix and features given a dictionary of entities and a list of relationships on relational tables. \\
        For single Dataframe, we start with dataset Normalization.

        Methods:
        ------------
        generate([dataframe,levels,...]) : generate new features using deep feature synthesis

        Example :
        -----------------
        >>> from feature_generation.generation import FGeneration
        >>> import pandas as pd
        >>> #Generate a dataframe with random values (1000,6)
        >>> d = {'A': random.sample(range(0, 1000), 1000),
        ...     'B': random.sample(range(0, 1000), 1000),
        ...     'C': random.sample(range(0, 1000), 1000),
        ...     'D': random.sample(range(0, 1000), 1000),
        ...     'E': random.sample(range(0, 1000), 1000),
        ...     'F': random.sample(range(0, 1000), 1000)}
        >>> df = pd.DataFrame(d, columns = ['A', 'B','C','D','E','F']) 
        >>> df1 ,f= FGeneration.generate (dataframes=[df],Target='F',levels=3)    
        

    """
    def __init__(self):
        pass

    @classmethod
    def generate(cls,dataframes: 'list([pd.Dataframe,pd.Dataframe,...])' =[],
                      target_entity: str =None,
                      relations: 'dict[str -> list[int,int]]' =None,
                      Entities_names: 'list[(str, str, str, str)]'=None,
                      Entityset_name: str =None,
                      max_depth: int =2,
                      levels: int =None,
                      Target: str =None,
                      skip_norm: bool =False,
                      add_NLP: bool=False,
                      add_all_primitives: bool=False)-> pd.DataFrame:
        """
        generate: Method
        ---------

        parameters :
        -----
        • dataframes (list[(pd.Dataframe, pd.Dataframe, pd.Dataframe)]) - The list of our Dataframes.\\
        • target_entity (str)                                           - ’Entity id’ of entity on which to make predictions. If None
                                                                            the target entity will be detected automatically.\\
        • Entities_name (list[(str, str, str, str)])                    - The names of our entities, the names will
                                                                            be affected to each dataframe in the same order as if dataframes.\\
        • Entityset_name (str)                                          - Entityset.
                                                                            Default: Entityset with id=’dummy name’\\
        • max depth (int)                                               - Maximum allowed depth of features.
                                                                            Default: 2\\
        • levels (int)                                                  - This is used only if our len(dataframes)==1.
                                                                            Default: dataframes[0].shape[1]\\
        • relations (dict[str -> list[int,int]])                        - List of relationships between entities. List
                                                                            items are a list with the format (index of parent entity in the dataframes list,index
                                                                            of child entity in the dataframes list). If None the functions will try to detect the
                                                                            relationships between our dataframes but it’s recommended to pass them to the
                                                                            function as a parameters.\\
        • skip norm (bool,Optional)                                     - If True we skip the Normaliztion process of our dataset,
                                                                            and This is used only if our len(dataframes)==1.
                                                                            Default: False\\
        • target (str)                                                  - The name of the target column\\
        •add_NLP (bool)                                                  - Add NLP techniques Default: False\\
        •add_all_primitives (bool)                                       - add all types of primitives Default: False

        Returns
        ----------
        Dataframe : New Dataframe with new generated columns

        """
        #perform DFS on a single Dataframe 
        if len(dataframes)==1:
            df,feats=SDFGeneration.generate(df=dataframes[0],Entityset_name=Entityset_name,max_depth=max_depth,levels=levels,Target=Target,skip_norm=skip_norm,
                                            add_all_primitives=add_all_primitives,add_NLP=add_NLP)
        #perform DFS on multiple dataframes
        elif len(dataframes)>1:
            df,feats,_=MDFGeneration.generate(dataframes=dataframes,target_entity=target_entity,relations=relations,Entities_names=Entities_names,
                                              Entityset_name=Entityset_name,max_depth=max_depth,add_NLP=add_NLP,add_all_primitives=add_all_primitives)
        else : 
            print(' You should provide a dataframe at least  ')
        return df,feats





