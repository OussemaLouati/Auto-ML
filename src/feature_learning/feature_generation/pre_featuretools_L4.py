import featuretools as ft
from .pre_featuretools_L2 import PreFeaturetoolsL2 as PFL2

class PreFeaturetoolsL4:
    """
        PreFeaturetoolsL4
        -------------
        Methods:
        --------
        Detect_target_entity([entitySet]): Return target entity in an entityset\\
        Add_relationships([relations, entityset,...]): Add relationships to an entityset
    
    """
    def __init__(self):
        pass

    @classmethod
    def Detect_target_entity(cls,entityset: ft.EntitySet =None)-> str:
        '''Detect target entity
        parameters 
         ----------
            • entityset (ft.entitySet) - entityset
        
         returns
         --------
            str : id of the target entity 
        '''
        #target entity is the entity with the most number of rows
        target_entity=entityset.entities[0].id
        #initialize max with the number of rows in the first entity
        max=entityset.entities[0].shape[0]
        for e in entityset.entities:
            if e.shape[0]> max: 
                target_entity=e.id
                max=e.shape[0]
        return target_entity

    @classmethod
    def Add_relationships(cls,relations=None,entityset=None,Entities_names=None):
        """ Add relationships to an entityset
        Parameters
        -------
        • Entities_name (list[(str, str, str, str)])                    - The names of our entities\\
        • Entityset (ft.entitySet)                                          - Entityset.\\
        • relations (dict[str -> list[int,int]])                        - List of relationships between entities.
        Returns
        -------
        EntitySet
        """
        for r in relations : 
            #get names of entities
            en_1= 'entity_'+str(relations[r][0]) if Entities_names==None else Entities_names[relations[r][0]]
            en_2= 'entity_'+str(relations[r][1]) if Entities_names==None else Entities_names[relations[r][1]]
            #create relationship
            new_relationship = ft.Relationship(entityset[en_1][r],
                                    entityset[en_2][r])
            #add relationship
            entityset = entityset.add_relationship(new_relationship)
        return entityset 
