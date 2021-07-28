import pandas as pd
import numpy as np

class Scaler():
    def __init__(self):
        pass

    def __init__(self,columns=None,scaler=None): 
        """
         Parameters
        ------------
           • columns (List(String)) - Contains the names of the columns.
           • Scaler (Sklearn scaler object) - Takes an Sklearn scaler object such as MinMaxScaler().
        """ 
        self.columns=columns 
        self.scaler=scaler
        self.fitted=False


    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs:32-convert-Dpreprocessing-toSklearn-transformers free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        if((type(X) != pd.SparseDataFrame) and
           (type(X) != pd.DataFrame)):
            raise ValueError("X must be a DataFrame")
        assert isinstance(self.columns, list)
        self.fitted=True
        return self

    def transform(self, X, y=None, **kwargs):
        """
        an abstract method that is used to transform according to what happend in the fit method
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        if self.fitted:
            X[self.columns]=X[self.columns].astype("float")
            #Scaling
            for column in self.columns :
                x = X.loc[:, column].values.reshape(-1,1)
                X[column] = self.scaler.fit_transform(x)
            return X
        else : 
            raise ValueError('You should call Fit function Before Transform')

    def fit_transform(self, X, y=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        self = self.fit(X, y)
        return self.transform(X, y)