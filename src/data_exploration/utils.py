import json
from collections import OrderedDict, namedtuple

class Utils():
    def __init__(self,dicte):
        """
        Parameters:
        ---------------
            • json path (str) - Path to the Json file of annotation and Data Types
            
        methods: 
        --------------
            •GetTypeByColumnName(columnName): 
                Return the type of the column name

            •GetListOfAnnotations(): 
                Return namedTuple of Annotations

            •GetListOfTypes():
                Return namedTuple of types

            •plot([]): 
                Display the plot

            •color(): 
                @classmethod used to generate a color randomly

         Example :
        -----------------
        >>> from data_exploration.utils import Utils
        >>> import pandas as pd

        >>> #Read the Json file for annotations and data Types
        >>> Utils=Utils('example.json')

        >>> #Get List of Annotations and list of Data Types
        >>> Annotations=Utils.GetListOfAnnotations()
        >>> Types=Utils.GetListOfTypes()

        >>> #Unpacking the Named Tuples
        >>> kpis,geos,Datetime=Annotations
        >>> Numerical_Categorical,Numerical_Continuous,Numerical_High_cardinality, Text_High_cardinality,Text_Categorical=Types
        
        >>> #List of All Numerical Columns
        >>> Numerical=Numerical_Categorical+Numerical_Continuous+Numerical_High_cardinality
        """
        self.data= OrderedDict(dicte)
        
    def GetTypeByColumnName(self,columnName: str)->str:
        return self.data[columnName][1]['description']

    def GetListOfAnnotations(self)-> namedtuple:
        #We are using namedTuple because they are the most convinient way to store data
        #you can see namedTuple as Class that only stores attribute with no methods
        kpis=[]
        geos=dict()
        datetime=[]
        #Initialize a namedTuple of name 'Annotations' with three attributes 
        # 'kpis' : (list of kpis) , geos : dictionary of geo data , datetime :  list of datetime columns
        Annotations = namedtuple('Annotations', ['kpis', 'geos','Datetime'])
        #iterate our dictionary (parsed from the json file)
        for d in self.data :
            #ignore the column index ( this column was added by speedml by default)
            if d=='index' or d=='level_0':
                continue
            e=1 if 'annotation' in self.data[d][1] else 2
            if self.data[d][e]['annotation']=='KPI':
                 kpis.append(d)
            elif self.data[d][e]['annotation']=='Latitude':
                 geos['Latitude']=d
            elif self.data[d][e]['annotation']=='Longitude':
                 geos['Longitude']=d
            elif self.data[d][e]['annotation']=='Datetime' or self.data[d][e]['annotation']=='Date':
                 datetime.append(d)
            else:
                continue
        annotations=Annotations(kpis,geos,datetime)    
        return annotations
    
    def GetListOfTypes(self)->namedtuple:
        Numerical_Categorical=[]
        Text_Categorical=[]
        Text_High_cardinality=[]
        Numerical_High_cardinality=[]
        Numerical_Continuous=[]
        Types = namedtuple('Types', ['Numerical_Categorical','Numerical_Continuous','Numerical_High_cardinality', 'Text_High_cardinality','Text_Categorical'])
        for d in self.data :
            if d=='index' or d=='level_0':
                continue
            if 'description' in self.data[d][1]:
                if self.data[d][1]['description']=='Numerical Categorical':
                    Numerical_Categorical.append(d)
                elif self.data[d][1]['description']=='Numerical Continuous':
                    Numerical_Continuous.append(d)
                elif self.data[d][1]['description']=='Numerical High-cardinality':
                    Numerical_High_cardinality.append(d)
                elif self.data[d][1]['description']=='Text Categorical':
                    Text_Categorical.append(d)
                elif self.data[d][1]['description']=='Text High-cardinality':
                    Text_High_cardinality.append(d)
                else:
                    continue
        types=Types(Numerical_Categorical,Numerical_Continuous,Numerical_High_cardinality,Text_High_cardinality,Text_Categorical)    
        return types
    
    @classmethod
    def color(cls):
        import matplotlib, random
        hex_colors_dic = {}
        rgb_colors_dic = {}
        hex_colors_only = []
        for name, hex in matplotlib.colors.cnames.items():
            hex_colors_only.append(hex)
            hex_colors_dic[name] = hex
            rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)
        return random.choice(hex_colors_only)
