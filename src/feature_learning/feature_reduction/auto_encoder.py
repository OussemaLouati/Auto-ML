from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
import numpy as np


class Autoencoder:
    """ 
        Autoencoder 
        ------------
        Build an Autoencoder using Keras 

        Methods  
        ------------
        build([inputDim, latentDim,...]): Build an AutoEncoder
    """

    def __init__(self):
        pass

    @classmethod
    def build(cls,inputDim: int =None, latentDim: int =None):
        '''Parameters
        -----
        • inputDim (int) -  Input dimension\\
        • latentDim (int) - Compressed dimension\\

        Returns
        -----------
        Encoder, Autoencoder
        '''
        # initialize shape and define the input to the encoder
        input_df = Input(shape=(inputDim,))
        # build the encoder model
        latent_vector = Dense((latentDim+inputDim)//2, activation='relu')(input_df)
        latent_vector = Dense(latentDim, activation='relu',name='latent_vector')(latent_vector)
        # start building the decoder model which will accept the
		# output of the encoder as its inputs
        y = Dense((latentDim+inputDim)//2, activation='relu')(latent_vector)
        outputs= Dense(inputDim, activation='sigmoid')(y)
        # build the decoder model
        encoder = Model(input_df, latent_vector, name="encoder")
        # build the autoencoder model
        autoencoder = Model(input_df, outputs,name="autoencoder")
        return encoder,  autoencoder
