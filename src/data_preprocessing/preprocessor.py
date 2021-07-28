import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    This class deals with imputing and dropping missing values.
    """
    def __init__(self, preprocessor_type={}, on_strategy='mean',percentage = 90.00):
        """
        Parameters
        ------------
           • preprocessor_type (Set(String)) - Take a one or many preprocess methods to apply (impute, drop, dropColumns).
           • strategy (string) - Dedicated for the imputing strategy.
           • percentage (float) - Percentage of missing values in columns to be deleted (for the dropColumns method).
        """
        self.preprocessor_type=preprocessor_type
        self.on_strategy=on_strategy
        self.percentage = percentage
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
        print(type(self.preprocessor_type))
        if self.fitted:
            for preprocessor_type in self.preprocessor_type:   
                if  preprocessor_type=="drop" : 
                    initial_row = X.shape[0]
                    empty_target_rows = y.isnull()
                    X = X.drop(X[ empty_target_rows ].index, axis=0)
                    print (str(empty_target_rows.sum())+" empty target rows has been deleted from "+ str(initial_row)+".")
                elif preprocessor_type=="impute":
                    categorial_features = X.columns[(X.dtypes == 'object')|(X.dtypes == 'categorical')]
                    numerical_features = [ x for x in X.columns if not x in categorial_features ]
                    #All nan values
                    nan_values = X.isnull().sum().sum()
                    #Fill nan values with most frequent value in categorical features
                    imputer_categorical = SimpleImputer(strategy="most_frequent")
                    X[categorial_features] = imputer_categorical.fit_transform(X[categorial_features])
                    #Fill nan values with the chosen strategy for numerical features
                    imputer_numerical = SimpleImputer(missing_values=np.nan, strategy=self.on_strategy)
                    X[numerical_features] = imputer_numerical.fit_transform(X[numerical_features])
                    print('Filled {} null values across the dataset.'.format(str(nan_values)))
                elif preprocessor_type=="dropColumns":
                    missing_val_percent = 100 * X.isnull().sum() / X.shape[0]
                    cols=[index for index,i in zip(missing_val_percent.index,missing_val_percent) if i >= self.percentage]
                    #drop columns
                    X.drop(cols,axis=1,inplace=True)
                    print(str(len(cols))+" column(s) were deleted \n"+ "There are now "+str(X.shape[1])+" column(s) in your dataframe.")
                else : 
                    raise ValueError(" {} ".format(preprocessor_type))

        else : 
            raise ValueError('You should call Fit function Before Transform')
        return X
        
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
   
    @classmethod
    def missing_values(cls,data):
        """
        Preview the number and % of the missing values in each columns.
            Returns:
            • Pandas dataframe
        """
        missing_vals = data.isnull().sum()
        missing_val_percent = 100 * missing_vals / data.shape[0]
        missing_vals_df = pd.concat( [missing_vals , missing_val_percent], axis=1)
        missing_vals_df = missing_vals_df.rename(columns = {0 : 'Missing_Values', 1 : '% of Missing values'})
        return missing_vals_df.sort_values(by="% of Missing values", ascending=False)

        
