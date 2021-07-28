import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
import plotly.express as px
from scipy import stats
import plotly.figure_factory as ff

class CorrelationMatrix(Plot):
    """
        Correlation Matrix of a certain Numerical columns.

        Example :
        -----------------
        >>> from data_exploration.correlationMatrix import CorrelationMatrix
        >>> import pandas as pd
        >>> CorrelationMatrix = CorrelationMatrix ()
        >>> d = {'A': [10,2,15,12],
        ...     'B': [2,4,5,1],
        ...     'C': [1,2,10,20]}
        >>> df = pd.DataFrame(d, columns = ['A', 'B','C'])     
        >>> CorrelationMatrix.trace(Dataframe = df, 
        ...                        columns=["A","C"]
        ...                        coef =’spearman ’)
        >>> CorrelationMatrix.plot()

        """
    def __init__(self):
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame, columns: 'list([str,str,...])' =[], coef: str ='pearson')-> go.Heatmap:
        """ 
        Parameters:
        ------------
            • Dataframe (pd.Dataframe)        - Our Dataframe\\
            • columns (list[str,str,str,str]) - List of Numerical Columns.\\
            • coef (str)                      - If coef is ’pearson’ we use pearson’s correlation coefficient and if coef is spearman we use the spearman’s one.
                                                Default: ’pearson’
        """
        if(len(columns)>1) :
            if coef=='pearson':
                #calculate the pearson coefficients
                temp=[[stats.pearsonr(Dataframe[x], Dataframe[y]) for y in columns] for x in columns]
                corr=[[corr for corr,_ignore in temp[i] ]for i in range(0,len(columns))]
            elif coef=='spearman':
                #calculate the spearman coefficients
                corr, _ = stats.spearmanr(Dataframe[columns], Dataframe[columns])
            trace = go.Heatmap(x=columns , 
                y= columns, 
                z= corr  ,
                colorscale= "Viridis",                
                     )
            #add trace
            self.List_traces.append(trace) 
            #add title   
            self.title= 'Correlation Matrix using ' + coef + ' correlation coefficients '
        else:
            print('You need to Provide at least 2 columns')
            return False                