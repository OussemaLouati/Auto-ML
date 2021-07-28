import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin


class Selection(BaseEstimator, TransformerMixin):
    def __init__(self,filter='CORR',skip_target=False,threshold=None):
        ''' 
        Parameters:
        ------
        
        • filter (str)                  - Type of filter to use : "CORR" for High correlation filter and \\
                                          "VAR" for low variance filter , Default: "CORR"\\
        • threshold (int)               - if filter == "CORR", The threshold value indicating high correlation.\\
                                            Default: 0.5.\\
                                         if filter=="VAR",-We will only return the variables that have a variance greater than our threshold.\\
                                          Default: 10\\
        • skip_target (bool,Optional)   - If True! we will skip the step of starting by keeping
                                            only the columns that are highly correlated with the target and then proceed with
                                            the pair wise correlation calculation. You should note also that even if this option
                                            is set to False and the calculation of the correlations of our columns with the target
                                            columns returned a list([’target’]) , only then this step will be skipped automatically
                                            because the function will then return just a dataframe with only a target column.\\
                                            Default: False
        '''
        self.skip_target=skip_target
        self.fitted=False 
        self.target=None
        self.filter=filter
        self.df=None
        self.threshold=threshold
        if self.filter== "CORR" and self.threshold is None : 
            self.threshold=0.5
        elif self.filter== "VAR" and self.threshold is None : 
            self.threshold=10
        
       

    def fit(self, X, y, **kwargs):
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
        if y is not None:
            if (type(y) != pd.core.series.Series):
                raise ValueError("y must be a Series")
            df=pd.concat([X,y],axis=1)
            self.df=df
            intersect=[x for x in list(df.columns) if x not in list(X.columns)]
            self.target=intersect[0]
        if self.filter not in ['VAR','CORR']:
            raise ValueError("{} not a valid filter".format(self.filter))
        
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
            
            if self.filter=='VAR':
                var = X.var()
                numeric = X.columns
                variables = [ ]
                for i in range(0,len(var)):
                    if var[i]>=self.threshold :
                        variables.append(numeric[i])
                X[variables]
            else : 
                cor = self.df.corr()
                cor_target = abs(cor[self.target])
                relevant_features = cor_target[cor_target>=self.threshold]
                if ((len(relevant_features)<=1) or (self.skip_target==True)):
                    relevant_features=cor_target
                d_corr=dict(relevant_features)
                relevant_column_names=list(d_corr.keys())
                relevant_column_names.remove(self.target)
                if(len(relevant_column_names)==0):
                    relevant_column_names=list(self.df.columns)
                i=1
                final_selection=d_corr.copy()
                for d in final_selection : 
                    final_selection[d]=1
            
                for col in relevant_column_names[:len(relevant_column_names)-1]:
                    if final_selection[col]==0 : 
                        i=i+1
                        continue 
                    else : 

                        for c in relevant_column_names[i:]:
                            if final_selection[c]==0 : 
                                continue
                            else: 
                                corr_val=abs(self.df[[col,c]].corr()[col][c])
                                if corr_val > self.threshold : 
                                    feature_to_discard= col if d_corr[col]<d_corr[c] else c 
                                    final_selection[feature_to_discard]=0
                        i=i+1
                cols=[d for d in final_selection if final_selection[d]==1 ]
                cols.remove(self.target)
                return X[cols]
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

    