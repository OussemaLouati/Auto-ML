import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
from .utils import Utils

class Kpi(Plot):
    """
        Line chart of our Kpi columns over time.
        
        Example :
        -----------------
        >>> from data_exploration.kpi import Kpi
        >>> import pandas as pd
        >>> Kpi = Kpi()
        >>> d = {'kpi1': [50.0,117.0,112.5,16.17],
        ...     'kpi2': [2.5,114,115.2,213.4],
        ...      'date':['3/11/2000', '3/12/2000', '3/13/2000','3/14/2000', '3/15/2000']}
        >>> df = pd.DataFrame(d, columns = ['kpi1', 'kpi2','date'])     
        >>> Kpi.trace(Dataframe = df, 
        ...                  kpis=["kpi2"]
        ...                  time_column="date")
        >>> Kpi.plot()

        """
    def __init__(self):
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame, kpis=[],time_column: 'DateTime column'=None)-> go.Scatter:  
        """
        Parameters
        ---------
            • Dataframe (pd.Dataframe)     - Our Dataframe\\
            • kpis (list[str,str,str,str]) - List of our Kpis. our kpis will be in the same chart (
                                            stacked line chart), and if len(kpis)==1 , this will plot our kpi over time with 3 other lines of the average over
                                             7/30/90 periods.\\
            • time column (pd.Datetime) - Datetime column. 
        """   
        if((len(kpis))==0 or (time_column==None)) :
            print('You must Specify at least a KPI and a Time Column') 
            return False
        else:
            if (len(kpis)==1):
                Dataframe = Dataframe.sort_values(by=time_column)
                temp=Dataframe.groupby([time_column])[kpis[0]].count().reset_index()
                #create a list of the average of every 7 values of our kpi  
                temp['average of 7']=temp.rolling(window=7, center=True)[kpis[0]].mean()
                #create a list of the average of every 30 values of our kpi  
                temp['average of 30']=temp.rolling(window=30, center=True)[kpis[0]].mean()
                #create a list of the average of every 90 values of our kpi  
                temp['average of 90']=temp.rolling(window=90, center=True)[kpis[0]].mean()
                #add 4 traces of our kpi and the averages of 7/30/90 over time , each line will have a random column 
                trace1=go.Scatter(x=temp[time_column], y=temp[kpis[0]], name=kpis[0] ,mode='lines',line = dict(color=Utils.color(), width=2))
                trace2=go.Scatter(x=temp[time_column], y=temp['average of 7'], name='average of 7' ,mode='lines',line = dict(color=Utils.color(), width=2))
                trace3=go.Scatter(x=temp[time_column], y=temp['average of 30'], name='average of 30' ,mode='lines',line = dict(color=Utils.color(), width=2))
                trace4=go.Scatter(x=temp[time_column], y=temp['average of 90'], name='average of 90' ,mode='lines',line = dict(color=Utils.color(), width=2))
                #list of traces
                data=[trace1, trace2, trace3, trace4]
                self.List_traces=data
                #add label to y axix
                self.y_label=kpis[0]
                #add title
                self.title='Line Chart of '+kpis[0]+' over  '+time_column
            else:
                Dataframe = Dataframe.sort_values(by=time_column)
                #iterate through all kpis and plot a stacked line chart of all kpis
                for kpi in kpis:
                    trace=go.Scatter(x=Dataframe[time_column], y=Dataframe[kpi], name=kpi ,mode='lines',line = dict(color=Utils.color(), width=2))
                    self.List_traces.append(trace)    
                #add title
                self.title= 'Stacked Line Chart of KPI\'s over  ' + time_column
            #label of x axis
            self.x_label= 'Date'