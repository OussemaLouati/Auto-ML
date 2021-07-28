import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
import plotly.express as px

class Relation(Plot):
    """
        Plot relationship betwwen two (Scatter plot) or three columns (Bubble plot)
        
        Example :
        -----------------
        >>> from data_exploration.relation import Relation
        >>> import pandas as pd
        >>> Relation = Relation()
        >>> d = {'A': [50.0,117.0,112.5,16.17],
        ...     'B': [2.5,114,115.2,213.4],
        >>> df = pd.DataFrame(d, columns = ['A,'B'])     
        >>> Relation.trace(Dataframe = df, 
        ...          x='A'
        ...          y='B')
        >>> Relation.plot()

        """
    def __init__(self):
        super().__init__() 
    
    def trace(self,Dataframe: pd.DataFrame, x: 'Numerical Column name' =None, y: 'Numerical Column name' =None, z: 'Numerical Column name' =None)->go.Scatter:
        """ 
        Parameters
        ------------
            • Dataframe (pd.Dataframe) - Our Dataframe\\
            • x (str) - Name of the first column. Should be of a numerical type.\\
            • y (str) - Name of the second column. Should be of a numerical type.\\
            • z (str) - Name of the third column. Should be of a numerical type. if None this will
                        display a scatter plot if not none this column will be considered as the size of the
                        point and therefore this will display a bubble plot.
        """
        #we should at least specify x and y
        if((x!=None) and (y!=None)):
            #if z is defined it will considered as the size of the dot 
            size=Dataframe[z] if z!=None else None
            #if z is defined we should display a 'showscale in our plot that's why we defined b 
            b=True if z!=None else False
            #add title 
            self.title= x + ' V. ' + y if z==None else x + ' V. ' + y + ' V. ' + z
            trace=go.Scatter(
                        x=Dataframe[x],
                        y=Dataframe[y],
                        mode='markers', 
                        marker=dict(size=size,color=size,showscale=b)
                        )
            self.List_traces.append(trace)    
            self.x_label= x
            self.y_label= y
        else:
            print('You must at least specify X and Y')
            return False

