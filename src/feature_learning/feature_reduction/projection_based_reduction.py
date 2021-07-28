from ..utils import Utils
import pandas as pd
from sklearn.decomposition import KernelPCA
from sklearn.manifold import TSNE 
from keras.optimizers import Adam
import numpy as np
from .auto_encoder import Autoencoder
from sklearn.base import BaseEstimator, TransformerMixin
        

class ProjectionBasedReduction(BaseEstimator, TransformerMixin):
    """ 
        ProjectionBasedReduction: Feature Extraction module using non-Linear Techniques 
        ----------- 
        AutoEncoders,
        T-sne,
        KernelPCA.

        Example 
        -----------------
        >>> from feature_reduction.projection_Based_Reduction import ProjectionBasedReduction
        >>> import sklearn.model_selection as model_selection
        >>> import pandas as pd
        >>> import random
        >>> #Generate a dataframe with random values (1000,6)
        >>> d = {'A': random.sample(range(0, 1000), 1000),
        ...     'B': random.sample(range(0, 1000), 1000),
        ...     'C': random.sample(range(0, 1000), 1000),
        ...     'D': random.sample(range(0, 1000), 1000),
        ...     'E': random.sample(range(0, 1000), 1000),
        ...     'F': random.sample(range(0, 1000), 1000)}
        >>> df = pd.DataFrame(d, columns = ['A', 'B','C','D','E','F']) 
        >>> # train test split
        >>> X=df.drop(columns =['F'])
        >>> y=df['F']
        >>> X_train , X_test , y_train , y_test = model_selection.train_test_split (X,y, train_size =0.65 , test_size =0.35 , random_state =101) 
        >>> # Perform a dimensionality reduction using autoencoders
        >>> Proj = ProjectionBasedReduction(technique='AENCODER', n_features=4, )
        >>> newdf= Proj.fit_transform(X_train,  y_train )
      

        """
    def __init__(self,  technique='KPCA',
                        epochs: int =20,
                        batch_size: int =500,                      
                        n_features: int =None,
                        kernel: str ='rbf'):
        ''' 
        Using a feature Extraction Technique

        Parameters
        -----
        • technique (str)   - "KPCA" for Kernel PCA \\
        ...                     "TSNE" for t-distributed stochastic neighbor embedding \\
        ...                     "AENCODER" for denoising autoencoder\\
        • n features (int) - Number of feature we want to have after performing our reduction.\\
        • latentDim (int) - The size of the latent vector ( Bottleneck ), i.e the number of features we want to keep\\
        • epochs (int) - Number of epochs to use in training\\
                        Default: 20\\
        • batch size (int) - The size of batches
                            Default: 250\\
        • kernel (str) - “linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”, used with "KPCA"\\
                        Default=”rbf”.

        Returns
        -----------
        Dataframe : a dataframe number of coolumns = latentDim
        
        '''
        self.technique=technique
        self.epochs=epochs
        self.batch_size=batch_size
        self.n_features=n_features
        self.kernel=kernel
        self.fitted_instance=None
        self.fitted=False
        if self.technique=='KPCA' : 
            self.transformer = KernelPCA(n_components=n_features, kernel=kernel)
        elif self.technique=='TSNE':
            self.transformer = TSNE(n_components=n_features)
        elif self.technique=='AENCODER':
            self.encoder=None
            self.autoencoder=None
        else : 
            raise ValueError('{} is not a valid projection based technique'.format(self.technique))
   
    def fit(self, X, y=None,X_val=None,y_val=None, **kwargs):
        """
        an abstract method that is used to fit the step and to learn by examples
        :param X: features - Dataframe
        :param y: target vector - Series
        :param X_val: features - Dataframe used to train the autoencoder
        :param y_val: target vector - Series used to train the autoencoder
        :param kwargs: free parameters - dictionary
        :return: self: the class object - an instance of the transformer - Transformer
        """
        if((type(X) != pd.SparseDataFrame) and
           (type(X) != pd.DataFrame)):
            raise ValueError("X must be a DataFrame")
        if (type(y) != pd.core.series.Series):
            raise ValueError("y must be a Series")
        if self.n_features is None : 
            raise ValueError("n_features is not set.")
        if self.technique in ['KPCA','TSNE']:
            self.fitted_instance=self.transformer.fit(X)
        else:
            #input shape 
            inputDim=X.shape[1]
            # add noise to X_train
            X_train_noisy = X + np.random.normal(loc=0.0, scale=1.0, size=X.shape)
            X_train_noisy = np.clip(X_train_noisy, 0., 1.)
            # add noise _val
            X_val_noisy = X_val + np.random.normal(loc=0.0, scale=1.0, size=X_val.shape) if X_val !=None else None
            X_val_noisy = np.clip(X_val_noisy, 0., 1.) if X_val !=None else None
            # Construct our Autoencoder
            self.encoder,self.autoencoder=Autoencoder.build(inputDim,self.n_features)
            opt = Adam(lr=1e-3)
            self.autoencoder.compile(optimizer=opt, loss='mean_squared_error')
            # train the autoencoder
            self.autoencoder.fit(X_train_noisy, X, epochs=self.epochs, batch_size=self.batch_size, shuffle=True, validation_data=( X_val_noisy, X_val) if X_val !=None else None,verbose=0)  
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
        if self.fitted : 

            if self.technique in ['KPCA','TSNE'] :
                return self.fitted_instance.transform(X)
            else :
                #perform Feature reduction 
                encoded_X_train = self.encoder.predict(X)
                #encoded_X_train = Utils.Array_to_dataframe(Array=encoded_X_train, df_y=y)
                return encoded_X_train
        else : 
            raise ValueError('You should call Fit function Before Transform')

    def fit_transform(self, X, y=None,X_val=None,y_val=None, **kwargs):
        """
        perform fit and transform over the data
        :param X: features - Dataframe
        :param y: target vector - Series
        :param kwargs: free parameters - dictionary
        :return: X: the transformed data - Dataframe
        """
        self = self.fit(X, y,X_val,y_val)
        return self.transform(X, y)
