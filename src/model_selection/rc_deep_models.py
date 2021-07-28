from keras.layers import Activation, Dense, Input, Dropout
from keras.models import Model, Sequential
from keras import regularizers 
import keras.optimizers 
from .constants import KERAS_METRICS as km
import os
import sys
import math 
from .utils import Utils

class KerasModels:
    def __init__(self):
        pass
    @classmethod
    def build(cls, num_cols=None, optimizer='Adadelta', dropout=0.2, 
             kernel_initializer='normal', activation='elu',metric=None
             , final_activation='sigmoid',min_units=10,problem_type=None, 
             learn_rate=0.01,hidden_layers=(1, 0.75, 0.25)):

        ad_act=["LeakyReLU","PReLU","ELU","ParametricSoftplus","ThresholdedReLU","SReLU"]
        act_in_adv =activation in ad_act
        f_act_in_adv =final_activation in ad_act
        if act_in_adv :
            act = Utils._import("keras.layers.advanced_activations."+activation)()
        if f_act_in_adv:
            f_act = Utils._import("keras.layers.advanced_activations."+final_activation)()
        opt=Utils._import("keras.optimizers."+optimizer)(clipnorm=1.,lr=learn_rate)
        _layers = []
        for layer in list(hidden_layers):
            _layers.append(min(math.ceil(num_cols * layer), min_units))
        model = Sequential()
        model.add(Dense(_layers[0], input_dim=num_cols, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        activation= act if act_in_adv else Activation(activation)
        model.add(activation)
        for layer_size in _layers[1:-1]:
            model.add(Dense(layer_size, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
            model.add(activation)
            model.add(Dropout(dropout))
        model.add(Dense(_layers[-1], kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l2(0.01)))
        model.add(activation)
        if problem_type.split('_')[-1]=="classification":
            model.add(Dense(1, kernel_initializer=kernel_initializer, activation=f_act if act_in_adv else final_activation))
        else:
            model.add(Dense(1, kernel_initializer=kernel_initializer))
        loss = 'binary_crossentropy' if problem_type=="classification" else 'mean_squared_error'
        metrics = Utils._import("keras.metrics."+km[metric])()
        model.compile(loss=loss, optimizer=opt, metrics=[metrics])
        return model


