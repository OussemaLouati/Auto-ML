import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot

class Compare(Plot):
    """
        Create a Comparison Plot

        Example :
        -----------------
        >>> from data_exploration.compare import Compare
        >>> import pandas as pd
        >>> Compare = Compare ()
        >>> d = {'A': ['a','b','c','d'],
        ...     'B': [2,4,5,1]}
        >>> df = pd.DataFrame(d, columns = ['A', 'B'])     
        >>> Compare.trace(Dataframe = df, 
        ...                  x="B"
        ...                  y="A"
        ...                  type = 'box')
        >>> Compare.plot()

        """
    def __init__(self):
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame, x:'Numerical column'=None, y:'Ctegorical column'=None, type:str='Bar')-> 'bar/box plot':
        """
        Parameters:
        ------
            • Dataframe (pd.Dataframe) - Our Dataframe
            • x (str)                  - Name of the first column. Should be of a numerical type.
            • y (str)                  - Name of the second column, should be of a Categorical type.
            • type (str)               - If type is ’box’ then we will display Distribution of x for each Categoy  of y. 
                                         And If type is ’bar’ then we will display Sum of x for each Categoy of y Default: ’bar’

        """
        if((x!=None) and (y!=None)):
            if type=='Bar': 
                group=Dataframe.groupby([y])
                temp_df=group[x].count().reset_index().sort_values(by=x, ascending=False)
                trace=go.Bar(
                    x=temp_df[y],
                    y=temp_df[x],
                    text=temp_df[x],
                    textposition='outside',  
                )
                #add trace to list of traces
                self.List_traces.append(trace)
                self.title= 'Sum of ' + x + ' for each Categoy of  ' + y
                self.x_label=y
                self.y_label=x
            elif type=='Box':
                #categories names
                labels=Dataframe[y].unique()
                #iterate through the names to create a box plot for each distribution of x for that column name
                for label in labels:
                    df=Dataframe[Dataframe[y] == label]
                    trace=go.Box(x=df[x], boxpoints='all', jitter=0.3, pointpos=0,name=label )
                    self.List_traces.append(trace)
                self.title= 'Distribution of ' + x +' for each Categoy of  ' + y
                self.x_label=x
                self.y_label=y    
            
        else : 
            print('You must Specify x and y ')  
            return False  
    
    
        
    



    