import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd 
from .plot import Plot
import plotly.express as px

class Geo(Plot):
    """
        Create a Scatter plot on a map 

        Example :
        -----------------
        >>> from data_exploration.geo import Geo
        >>> import pandas as pd
        >>> Geo = Geo ()
        >>> d = {'lat': [50.2,117.3,112.5,16.17],
        ...     'long': [2.5,114,115.2,213.4]}
        >>> df = pd.DataFrame(d, columns = ['lat', 'long'])     
        >>> Geo.trace(Dataframe = df, 
        ...                  lat="lat"
        ...                  long="long")
        >>> Geo.plot()

        """
    def __init__(self):    
        super().__init__()
        
    def trace(self,Dataframe: pd.DataFrame,longitude: str =None, latitude: str =None)->go.Scattergeo:
        """
        Parameters
        ------
            • Dataframe (pd.Dataframe) - Our Dataframe\\
            • lat (float) - Latitude\\
            • long (float) - Longitude
        """
        if((longitude!=None) and (latitude!=None)):
            trace=go.Scattergeo(lon = Dataframe[longitude],
                            lat = Dataframe[latitude],      
                            mode = 'markers')
            self.List_traces.append(trace)    
            self.title='Scatter plot on a map based on the (longitude, Latitude ) Columns '
        else:
            print('You need to Provide both Longitude and Latitude Columns')
            return False





    

    



    