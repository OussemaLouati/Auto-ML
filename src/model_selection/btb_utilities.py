from btb.selection import (
     Uniform,UCB1, BestKReward, BestKVelocity, PureBestKVelocity, RecentKReward)
from btb.tuning import GPTuner, GPEiTuner, UniformTuner
from .method import Algorithm
from btb.tuning import GPTuner, Tunable
from .utils import Utils

class BtbUtils:
    TUNERS = {
    'uniform': UniformTuner,
    'gp': GPTuner,
    'gp_ei': GPEiTuner
    }

    SELECTORS = {
    'uniform': Uniform,
    'ucb1': UCB1,
    'bestk': BestKReward,
    'bestkvel': BestKVelocity,
    'purebestkvel': PureBestKVelocity,
    'recentk': RecentKReward,
    }

    def __init__(self):
        pass
    @classmethod   
    def prepare_candidates(cls,models=None):
        candidates={}
        for e in models:
            if type(e)==str:
                candidates[e]=Algorithm.get_model_by_name(model_name=e)
            else :
                if callable(e):
                    candidates[e.__name__]=e
                else :
                    candidates[e.__class__.__name__]=Utils._import(str(e.__class__))
        return candidates
    
    @classmethod
    def as_tunable(cls,hyperparams):
        return Tunable(hyperparams)
    
    @classmethod
    def load_tuner(cls,tuner='gp'):
        return cls.TUNERS[tuner]

    @classmethod
    def load_selector(cls,selector='ucb1'):
        return cls.SELECTORS[selector]

    @classmethod
    def to_btb_space(cls,space={}):
        btb_space={}
        for k,v in space.items():
            if type(v[0])== str :
                btb_space[k]=CategoricalHyperParam(choices=v, default=v[0])
            elif type(v[0])== int : 
                btb_space[k]=IntHyperParam(min=v[0], max=v[-1], default=v[0])
            elif type(v[0])== float : 
                btb_space[k]=FloatHyperParam(min=v[0], max=v[-1])
            elif type(v[0])== bool : 
                btb_space[k]=BooleanHyperParam(default=v[0])
            else : 
                raise ValueError(" {} : {} ".format(k,v))

        return btb_space

    
