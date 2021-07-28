import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
import plotly.express as px


class ScatterMatrix(Plot):
    """
        Example :
        -----------------
        >>> from data_exploration.ScatterMatrix import ScatterMatrix
        >>> import pandas as pd
        >>> ScatterMatrix = ScatterMatrix ()
        >>> d = {'A': [10,2,15,12],
        ...     'B': [2,4,5,1],
        ...     'C': [1,2,10,20]}
        >>> df = pd.DataFrame(d, columns = ['A', 'B','C'])     
        >>> ScatterMatrix.trace(Dataframe = df, 
        ...                    columns=["A","C","C"])
        >>> ScatterMatrix.plot()

        """
    def __init__(self):
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame ,columns: 'list([str,str,...])' =[])-> go.Splom:
        if(len(columns)!=0) :
            trace=go.Splom(
                dimensions=[ dict(label=c,values=Dataframe[c]) for c in columns],
                showupperhalf=False,
                diagonal_visible=False,
                marker=dict(showscale=False, 
                line_color='white', line_width=0.5,
                line=dict(width=0.5,color='rgb(230,230,230)'))
                )
            self.List_traces.append(trace)    
            self.title='Scatter Matrix '
        else:
            print('Columns cannot be empty')
            return False                