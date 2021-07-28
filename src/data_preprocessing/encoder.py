from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import category_encoders as ce
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class Encoder(BaseEstimator, TransformerMixin):
    """
    Encoder class has encoding methods for categorical columns.
    """
    def __init__(self,encoder="one_hot_encoder",columns_names=None,column_name=None,drop=True,suffix=None,ordinal_dict=None):
        """
         Parameters
        ------------
           • encoder (String) - For the type of encoding to be applied.
           • columns_names (List) - Contains the names of the columns which concern the encoding.
           • column_name(String) - Concerns the "one hot encoding" encoder column name.
           • drop(Boolean): - if True, the initial column will be deleted after the encoding. False, the initial column will stay in the dataframe.
           • suffix(String) -The suffix to be added in the columns names for the encoding "one hot encoding".
           • percentage (float) - Percentage of missing values in columns to be deleted (for the dropColumns method).
        """
        self.encoder=encoder
        self.columns_names=columns_names
        self.drop=drop  
        self.suffix=suffix   
        self.column_name=column_name  
        self.fitted=False
        self.ordinal_dict=ordinal_dict
        self.df=None
        self.target=None

        
    def fit(self, X, y=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        if((type(X) != pd.SparseDataFrame) and
           (type(X) != pd.DataFrame)):
            raise ValueError("X must be a DataFrame")
        assert(self.encoder in ["ordinal_encoder","binary_encoder","frequency_encoder","mean_encoder","helmert_encoder","one_hot_encoder","label_encoder"]), 'not a valid encoder method'
        if self.encoder == "ordinal_encoder":
            assert self.ordinal_dict != None
        if y != None :        
            if (type(y) != pd.core.series.Series):
                raise ValueError("y must be a Series")
            df = pd.concat([X,y],axis=1)
            intersect=[x for x in df.columns if x not in X.columns]
            self.target=intersect[0]
        else:
            df = X
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
            if self.encoder=="label_encoder":
                for col in self.columns_names :
                    fact = X[col].factorize()
                    X[col+"encoded"] = fact[0]
                if self.drop:
                    X.drop(self.columns_names,axis=1,inplace=True)
                return X
            elif self.encoder=="one_hot_encoder":     
                if self.suffix == None:
                    self.suffix = self.column_name + "_"
                #OneHotEncoder initiation
                ohe = OneHotEncoder()
                one = ohe.fit_transform(X.loc[:,[self.column_name]].values).toarray()
                #Add suffix for the new created features
                cols = [self.suffix + str(ohe.categories_[0][i]) for i in range(len(ohe.categories_[0]))]
                onehot_encoded_df = pd.DataFrame(one, columns = cols)
                if self.drop:
                    X.drop([self.column_name],axis=1,inplace=True)
                #return the new dataframe with the one_hot_encoded columns
                X=pd.concat([X,onehot_encoded_df],axis=1)
                return X
            elif self.encoder=="binary_encoder": 
                binary_encoder = ce.BinaryEncoder(cols=[self.column_name], drop_invariant=True)
                new_df = binary_encoder.fit_transform(X[self.column_name])
                if self.drop:
                    X.drop([self.column_name],axis=1,inplace=True)
                X= pd.concat([X, new_df], axis=1)
                return X
            elif self.encoder=="frequency_encoder": 
                frequency = X.groupby(self.column_name).size()/len(X)
                X["frequency_"+self.column_name] = X[self.column_name].map(frequency)
                return X
            elif self.encoder=="mean_encoder": 
                mean = self.df.groupby(self.column_name)[self.target].mean()
                self.df["mean_"+self.column_name] = self.df[self.column_name].map(mean)
                if self.drop:
                    self.df.drop([self.column_name],axis=1,inplace=True)
                return X
            elif self.encoder=="ordinal_encoder": 
                X[self.column_name] = X[self.column_name].map(self.ordinal_dict)
                if self.drop:
                    X.drop([self.column_name],axis=1,inplace=True)
                return X
            else : 
                helmert_encoder = ce.HelmertEncoder(cols=[self.column_name], drop_invariant=True)
                new_df = helmert_encoder.fit_transform(X[self.column_name])
                if self.drop:
                    X.drop([self.column_name],axis=1,inplace=True)
                X=pd.concat([X, new_df], axis=1)  
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