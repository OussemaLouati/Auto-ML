import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
from .utils import Utils
from plotly.subplots import make_subplots
import os

class Distribution(Plot):
    """Distribution of the different columns in our dataset.
        Parameters:
        --------
            • threshold (int)       -  If a column has less than this threshold categories we will display a pie chart else a bar chart.
                                       Default : 10 \\
            • maxCatgories (int)    -  Maximum numbers of categories to retain and plot if we have a column with a huge number of categories
                                       Default : 15\\

        Example :
        -----------------
        >>> from data_exploration.distribution import Distribution
        >>> import pandas as pd
        >>> Distribution = Distribution ()
        >>> d = {'A': ['a','b','c','d'],
        ...     'B': [2,4,5,1]}
        >>> df = pd.DataFrame(d, columns = ['A', 'B'])     
        >>> Distribution.trace(Dataframe = df, 
        ...                  columns =['B'],
        ...                  type = ’Numerical Categorical’)

        """
    def __init__(self,threshold=10,maxCatgories=15):
        self.threshold=threshold
        self.maxCatgories=maxCatgories
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame,columns:'list[str,str,str,...]'=[],type: str =None, auto_open: bool =True)-> 'Histogram/Bar/Pie/Box- Plot(s)':
        """Parameters
        ------------
            • Dataframe (pd.Dataframe)       - Our Dataframe\\
            • columns (lis[str,str,str,str]) - List of the columns to display, if len(columns)>1 the
                                               charts will be displayed as subplots.\\
            • type (str)                     -  Type of the column(s). Type should be one of these types :\\
                                               [’Numerical Categorical’,’Numerical Continuous’,’Numerical High-cardinality’,’Text Categorical’,’Text High-cardinality’]
        """
        if(len(columns)!=0 and type!= None) :
            i=1
            #initiliaze our subplots , subplots are of the form of a matrix, so to avoid Problems when when we have odd number of columns we crated our matrix as (?,1)
            subplots=make_subplots(len(columns),1,)
            #change the titles depending on how many columns we have if have just one column wa can omit the 'subplot' word 
            self.title= 'Subplots of Distribution of ' + type + ' Columns ' if len(columns)!=1 else 'Distribution of Column ' + columns[0] 
            #Histogram for Numerical Categorical
            if ((type == 'Numerical Categorical') ):
                for c in columns :
                    trace=go.Histogram(x=Dataframe[c],name=c,histnorm='probability')
                    #add this trace as a subplot in the position (i,1)  
                    subplots.append_trace(trace,i,1)
                    i=i+1
            #Box plot for 'Numerical High-cardinality' or  'Numerical Continuous'
            elif (type=='Numerical High-cardinality') or ( type == 'Numerical Continuous'):
                for c in columns :
                    trace=go.Box(x=Dataframe[c], boxpoints='all', jitter=0.3, pointpos=-1.8 if len(columns)==1 else 0,name=c )
                    #add this trace as a subplot in the position (i,1)  
                    subplots.append_trace(trace,i,1)
                    i=i+1
            elif type=='Text Categorical':
                if(len(columns)==1):
                    #Frequency of each category 
                    count=len(Dataframe[columns[0]].unique())
                    frequencies=(Dataframe[columns[0]].value_counts()).tolist()
                    #name of each category
                    labels=(Dataframe[columns[0]].unique()).tolist()
                    #test if different categories< threshold it will be a pie chart
                    if count<= self.threshold : 
                        #because all subplots should be of thesame type , here we re-initialize our subplots with type "pie"
                        subplots=make_subplots(len(columns),1,specs=[[{"type": "pie"}]])
                        trace=go.Pie(labels=labels,values=frequencies,textinfo='label+percent',insidetextorientation='radial')
                        subplots.append_trace(trace,1,1)
                    else:
                        #bar plot
                        trace=go.Bar(x=labels, y=frequencies,text=frequencies,textposition='auto',)
                        subplots.append_trace(trace,1,1)
                else:
                    #if we have multiple columns we will display a subplot of bar plots
                    for c in columns :
                        count=len(Dataframe[c].unique())
                        frequencies=(Dataframe[c].value_counts()).tolist()
                        labels=(Dataframe[c].unique()).tolist()
                        trace=go.Bar(x=labels, y=frequencies,text=frequencies,textposition='auto',name=c)
                        subplots.append_trace(trace,i,1)
                        i=i+1   
            #Bar plot for the most frequent categories ( we defined 15 categories) if  we have text High-Cardinality
            elif type=='Text High-cardinality':
                for c in columns :
                    #calculate Frequency of each category
                    frequencies=((Dataframe[c].value_counts())[:self.maxCatgories]).tolist()
                    #Extract name of columns for the most frequent categories
                    labels=((Dataframe[c].value_counts().index)[:self.maxCatgories]).tolist()
                    #add our trace
                    trace=go.Bar(x=labels, y=frequencies,text=frequencies,textposition='auto',name=c)
                    subplots.append_trace(trace,i,1)
                    i=i+1    
                #we overrride the title of the plot
                self.title='Subplots of Distribution of the ' +str(self.maxCatgories)+ ' most frequent categories of each ' + type + ' Columns ' if len(columns)!=1 else 'Distribution of the 15 most frequent categories of Column ' + columns[0] 
        else :
                print('Error : Type of Column' )
                return False  
        self.y_label= 'Counts'
        #Add Layout 
        subplots['layout'].update(title=self.title)
        if not os.path.exists('Data-Toolkit-Plots'):
            os.makedirs('Data-Toolkit-Plots')
        #Plot our Charts
        pyo.plot(subplots,filename='./Data-Toolkit-Plots/'+self.title+'.html',auto_open=auto_open)
     
        
    


