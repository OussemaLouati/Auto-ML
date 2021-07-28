from ..utils import Utils
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import FactorAnalysis as FA
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA  as ICA
from sklearn.base import BaseEstimator, TransformerMixin

class ComponentBasedReduction(BaseEstimator, TransformerMixin):
    """ 
        ComponentBasedReduction: Feature Extraction module using Linear Techniques 
        ------------
        Independent Component Analysis,
        Principal component analysis,
        Linear Discriminant Analysis,
        Factor Analysis.

        Example 
        -----------------
        >>> from feature_reduction.component_based_reduction import ComponentBasedReduction
        >>> import pandas as pd
        >>> #Generate a dataframe with random values (1000,6)
        >>> d = {'A': random.sample(range(0, 1000), 1000),
        ...     'B': random.sample(range(0, 1000), 1000),
        ...     'C': random.sample(range(0, 1000), 1000),
        ...     'D': random.sample(range(0, 1000), 1000),
        ...     'E': random.sample(range(0, 1000), 1000),
        ...     'F': random.sample(range(0, 1000), 1000)}
        >>> df = pd.DataFrame(d, columns = ['A', 'B','C','D','E','F'])     
        >>> reduction = ComponentBasedReduction(technique="PCA", n_features=2)
        >>> new_df = reduction.fit_transform(df)

    """
    Techniques = { "pca": PCA, "lda": LDA , "fa":FA, "ica":ICA}

    def __init__(self,technique : str =None,
                      n_features : int =None, 
                      pca_info_retain : "float : between 0 and 1"=0.95):

        """ 
        -------
        Parameters:
        ---------
        • technique (str)              - Name of the linear technique we want to use. the technique should be one from the following list : [’pca’,’lda’,’ica’,’fa’] , 
                                             note that here  the name of the technique is case insensitive.\\
        • n features (int)             - Number of feature we want to have after performing our reduction. \\
                                             Required if : technique = ’ICA’ or ’LDA’ or ’FA’ .\\
        • pca info retain (float)      - This will be considered only if we have n features=Noneand technique=’pca’\\
                                              Default: 0.95 , which means keeping a number of features than can preserve 95% of the underlying information in our data.\\
        
        Returns
        -----------
        Dataframe : a dataframe number of coolumns = n_features
        
        """
        self.technique=technique 
        self.n_features=n_features 
        self.pca_info_retain=pca_info_retain
        self.fitted=False
        self.fitted_instance=None
        if self.technique.lower() in self.Techniques : 
            self.tech = self.Techniques[self.technique.lower()](n_components=pca_info_retain if n_features==None and self.technique.lower()=='pca' else n_features )

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
        if y is not None :
            if (type(y) != pd.core.series.Series):
                raise ValueError("y must be a Series")
        if self.technique == 'lda':
            self.fitted_instance=self.tech.fit(X,y)
        else:
            self.fitted_instance=self.tech.fit(X)
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
            return self.fitted_instance.transform(X)
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
