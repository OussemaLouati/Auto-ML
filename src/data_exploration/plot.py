import plotly.offline as pyo
import plotly.graph_objs as go
import os
import tempfile
from abc import ABC, abstractmethod 
import pandas as pd

class Plot(ABC):
    def __init__(self):
        """Parameters:

            • title (str) - Title of the plot
            • x label (str) - Label of the x axis
            • y label (str) - Label of the y axis
            • List traces (list[plotly.graph objects, plotly.graph objects]) - Type of the graph to
            • Layout (plotly.graph_objects.Layout) - Layout of the plot.
            
            methods: 

            •layout([]): 
                Construct a new Layout object

            •figure([data, layout, ...]): 
                Create a new Figure instance

            •extract image([figure,name,...]):
                Extract the image of our plot in HTML

            •plot([]): 
                Display the plot

            •trace([dataframe,**kwargs]): 
                Define a plot, an @abstractmethod to be implemented by every derived class
            
            """
        self.title='' 
        self.x_label=''
        self.y_label=''
        self.Layout=None
        self.List_traces=[]
        
    @abstractmethod     
    def trace(self, Dataframe: pd.DataFrame, **kwargs:'Another key-word only arguments')->'trace':
        """Define a plot, an @abstractmethod to be implemented by every derived class"""
        return self.trace(Dataframe,**kwargs)

    def layout(self)->go.Layout:
        '''Construct a new Layout object'''
        return go.Layout(
                                title=self.title,
                                xaxis=dict(
                                        title=self.x_label,
                                        titlefont_size=16,
                                        tickfont_size=14),
                                yaxis=dict(
                                        title=self.y_label,
                                        titlefont_size=16,
                                        tickfont_size=14),
                               paper_bgcolor='rgb(243, 243, 243)',
                               plot_bgcolor='rgb(243, 243, 243)',
                               )
    def figure(self,data,layout):
        '''Create a new Figure instance'''
        return go.Figure(data=data, layout=layout)

    def extract_image(self,figure=None,name=None,auto_open=True):
        '''Extract the image of our plot in HTML'''
        if not os.path.exists('Data-Toolkit-Plots'):
            os.makedirs('Data-Toolkit-Plots')
        pyo.plot(figure, filename='./Data-Toolkit-Plots/'+name,auto_open=auto_open)
        
        
    def plot(self,auto_open=False):
        '''Display the plot'''
        #This is a wrapper method that uses the Layout and List of our traces to create and then plot our chart
        self.Layout=self.layout()
        fig=self.figure(self.List_traces, self.Layout)
        self.extract_image(fig, self.title+'.html',auto_open=auto_open)
        #We empty the list to avoid re-initialize our class every time we want to plot new chart
        self.List_traces=[]
   


    

