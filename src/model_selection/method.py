from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor, ExtraTreesRegressor, 
                                AdaBoostRegressor, GradientBoostingRegressor, GradientBoostingClassifier,
                                 ExtraTreesClassifier, AdaBoostClassifier)

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, 
                                    SGDRegressor, PassiveAggressiveRegressor, LogisticRegression, RidgeClassifier, 
                                    SGDClassifier,  PassiveAggressiveClassifier)
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC,SVR, LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

from btb.tuning.hyperparams import FloatHyperParam,IntHyperParam,CategoricalHyperParam, BooleanHyperParam

import numpy as np


class Algorithm :
    Classifiers = {
        'LogisticRegression': LogisticRegression,
        'RandomForestClassifier': RandomForestClassifier,
        'RidgeClassifier': RidgeClassifier,
        'GradientBoostingClassifier': GradientBoostingClassifier,
        'ExtraTreesClassifier': ExtraTreesClassifier,
        'AdaBoostClassifier': AdaBoostClassifier,
        'SGDClassifier' : SGDClassifier,
        'PassiveveAggressiveClassifier' : PassiveAggressiveClassifier,
        'SVC': SVC,
        'LinearSVC': LinearSVC,
        'BernoulliNB':BernoulliNB, 
        'GaussianNB':GaussianNB, 
        'MultinomialNB':MultinomialNB,
        'DecisionTreeClassifier':DecisionTreeClassifier,
        'KNeighborsClassifier': KNeighborsClassifier,
        'XGBClassifier' : XGBClassifier,
        'LGBMClassifier' : LGBMClassifier,
        'CatBoostClassifier' : CatBoostClassifier 
        }

    Regressors= {
        'LinearRegression': LinearRegression,
        'RandomForestRegressor': RandomForestRegressor,
        'Ridge': Ridge,
        'SVR':SVR,
        'LinearSVR': LinearSVR,
        'ExtraTreesRegressor': ExtraTreesRegressor,
        'AdaBoostRegressor': AdaBoostRegressor,
        'GradientBoostingRegressor': GradientBoostingRegressor,
        'Lasso': Lasso,
        'ElasticNet': ElasticNet,
        'SGDRegressor': SGDRegressor,
        'PassiveAggressiveRegressor'  :PassiveAggressiveRegressor,
        'DecisionTreeRegressor':DecisionTreeRegressor,
        'KNeighborsRegressor': KNeighborsRegressor,
        'LGBMRegressor' : LGBMRegressor,
        'XGBRegressor' : XGBRegressor,
        'CatBoostRegressor' : CatBoostRegressor

    }

    space = {
        'XGBClassifier': {
            'max_depth':[ [5,7,10,11,13,15,18,20,23,25,27,30,33,35,37,40,43,45,47,50], IntHyperParam(min=5, max=50, default=10)],
            'learning_rate':[ [0.01, 0.05, 0.1, 0.2], FloatHyperParam(min=0.01, max=0.5, default=0.05)  ],
            'n_estimators': [ [50, 75, 100, 150, 200, 375, 500, 750, 1000], IntHyperParam(min=50, max=2000, default=100)],
            'min_child_weight': [[1, 5, 10, 50], IntHyperParam(min=1, max=50, default=10)],
            'colsample_bytree': [[0.5,0.7, 0.8, 1.0] , FloatHyperParam(min=0.3, max=1.0, default=0.5)],
            'subsample': [[0.5,0.7,0.8, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)],
            'gamma': [[0.5,0.8, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)],
        },
        'XGBRegressor': {
            'max_depth': [ [5,7,10,11,13,15,18,20,23,25,27,30,33,35,37,40,43,45,47,50], IntHyperParam(min=5, max=50, default=10,step=1)],
            'booster': [['gbtree', 'gblinear', 'dart'], CategoricalHyperParam(choices=['gbtree', 'gblinear', 'dart'], default='gbtree')],
            #'objective': [['reg:squarederror', 'reg:gamma'], CategoricalHyperParam(choices=['reg:squarederror', 'reg:gamma'], default='reg:squarederror')],
            'n_estimators': [[50, 75, 100, 150, 200, 375, 500, 750, 1000], IntHyperParam(min=50, max=2000, default=100,step=1)],
            'min_child_weight': [[1, 5, 10, 50], IntHyperParam(min=2, max=50, default=10,step=1)],
            'colsample_bytree': [[0.5,0.7, 0.8, 1.0] , FloatHyperParam(min=0.3, max=1.0, default=0.5)],
            'subsample': [[0.5, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)],
            'learning_rate':[ [0.01, 0.05, 0.1, 0.2], FloatHyperParam(min=0.01, max=0.5, default=0.05)  ],
            'gamma': [[0.5,0.8, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)],


        },
        'GradientBoostingRegressor': {
            'max_depth': [[1, 2, 3, 4, 5, 7, 10, 15], IntHyperParam(min=1, max=30, default=10)],
            'max_features':[ ['sqrt', 'log2', None], CategoricalHyperParam(choices=['sqrt', 'log2', None], default='sqrt')],
            'loss': [['ls', 'huber'], CategoricalHyperParam(choices=['ls','huber'], default='ls')],
            'learning_rate':[ [0.01, 0.05, 0.1, 0.2], FloatHyperParam(min=0.01, max=0.5, default=0.05)  ],
            'n_estimators': [[10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000], IntHyperParam(min=50, max=2000, default=100,step=1)], #step=25
            'subsample': [[0.5, 0.65, 0.8, 0.9, 0.95, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)]
        },
        'GradientBoostingClassifier': {
            'max_depth': [[1, 2, 3, 4, 5, 7, 10, 15], IntHyperParam(min=1, max=30, default=10)],
            'max_features':[ ['sqrt', 'log2', None], CategoricalHyperParam(choices=['sqrt', 'log2', None], default='sqrt')],
            'loss': [['deviance', 'exponential'], CategoricalHyperParam(choices=['deviance', 'exponential'], default='deviance')],
            'learning_rate':[ [0.01, 0.05, 0.1, 0.2], FloatHyperParam(min=0.01, max=0.5, default=0.05)  ],
            'n_estimators': [[10, 50, 75, 100, 125, 150, 200, 500, 1000, 2000], IntHyperParam(min=50, max=2000, default=100)],
            'subsample': [[0.5, 0.65, 0.8, 0.9, 0.95, 1.0], FloatHyperParam(min=0.3, max=1.0, default=0.5)]
        },

        'LogisticRegression': {
            'C': [[.0001, .001, .01, .1, 1, 10, 100, 1000], FloatHyperParam(min=0.0001, max=1000)  ],
            'class_weight': [[None, 'balanced'], CategoricalHyperParam(choices=[None, 'balanced'], default='balanced')],
            'solver': [['newton-cg', 'lbfgs', 'sag'], CategoricalHyperParam(choices=['newton-cg', 'lbfgs', 'sag'], default='lbfgs')]
        },
        'LinearRegression': {
            'fit_intercept': [[True, False],BooleanHyperParam(default=True)],
            'normalize': [[True, False],BooleanHyperParam(default=False)]
        },
        'RandomForestClassifier': {
            'criterion': [['entropy', 'gini'], CategoricalHyperParam(choices=['entropy', 'gini'], default='entropy')],
            'class_weight': [[None, 'balanced'], CategoricalHyperParam(choices=[None, 'balanced'], default='balanced')],
            'max_features': [['sqrt', 'log2', None],CategoricalHyperParam(choices=['sqrt', 'log2', None], default='sqrt')],
            'min_samples_split': [[2, 5, 20, 50, 100],IntHyperParam(min=2, max=100, default=10,step=1)],#step=7
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100],IntHyperParam(min=1, max=100, default=10,step=1)], #step=3
            'n_estimators': [[50, 75, 100, 125, 150, 200,500], IntHyperParam(min=50, max=2000, default=100,step=1)], #step=25
            'max_depth': [[1, 2, 3, 4, 5, 7, 10, 15], IntHyperParam(min=2, max=30, default=10,step=1)] #step=2

        },
        'RandomForestRegressor': {
            'max_features': [['auto','sqrt', 'log2', None],CategoricalHyperParam(choices=['auto','sqrt', 'log2', None], default='auto')],
            'min_samples_split': [[2, 5, 20, 50, 100],IntHyperParam(min=2, max=100, default=10,step=1)], #step=7
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100],IntHyperParam(min=1, max=100, default=10,step=1)],  #step=3
            'n_estimators': [[50, 75, 100, 125, 150, 200,500], IntHyperParam(min=50, max=2000, default=100,step=1)], #step=10
            'max_depth': [[1, 2, 3, 4, 5, 7, 10, 15], IntHyperParam(min=2, max=30, default=10,step=1)] #step=2
        },
        'RidgeClassifier': {
            'alpha': [[.0001, .001, .01, .1, 1, 10, 100, 1000],FloatHyperParam(min=0.0001, max=1000)  ],
            'class_weight': [[None, 'balanced'], CategoricalHyperParam(choices=[None, 'balanced'], default='balanced')],
            'solver': [['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],CategoricalHyperParam(choices=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'], default='svd')]
        },
        'Ridge': {
            'alpha': [[.0001, .001, .01, .1, 1, 10, 100, 1000],FloatHyperParam(min=0.001, max=10)  ],
            'solver': [['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'],CategoricalHyperParam(choices=['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg'], default='svd')]
        },
        'ExtraTreesRegressor': {
            'max_features': [['auto','sqrt', 'log2', None],CategoricalHyperParam(choices=['auto','sqrt', 'log2', None], default='auto')],
            'min_samples_split': [[2, 5, 20, 50, 100],IntHyperParam(min=2, max=100, default=10,step=1)],#step=7
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100],IntHyperParam(min=1, max=100, default=10,step=1)], #step=3
        },
        'AdaBoostRegressor': {
            'loss': [['linear','square','exponential'],CategoricalHyperParam(choices=['linear','square','exponential'], default='linear')],
            'n_estimators': [[50, 75, 100, 125, 150, 200,500], IntHyperParam(min=50, max=2000, default=100,step=1)],#step=25
        },
        'ExtraTreesClassifier': {
            "criterion": [["entropy", "gini"],CategoricalHyperParam(choices=["entropy", "gini"], default='entropy')],
            'max_features': [['auto','sqrt', 'log2', None],CategoricalHyperParam(choices=['auto','sqrt', 'log2', None], default='auto')],
            'min_samples_split': [[2, 5, 20, 50, 100],IntHyperParam(min=2, max=100, default=10,step=1)],#step=7
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100],IntHyperParam(min=1, max=100, default=10,step=1)],#step=3
        },
        'AdaBoostClassifier': {
            'n_estimators': [[50, 75, 100, 125, 150, 200,500,1000], IntHyperParam(min=50, max=2000, default=100,step=1)], #step=25
        },
        'Lasso': {
            'selection': [['cyclic', 'random'],CategoricalHyperParam(choices=['cyclic', 'random'], default='cyclic')],
            'tol': [[.0000001, .000001, .00001, .0001, .001],FloatHyperParam(min=0.001, max=0.1)  ],
            'positive':[[True, False],BooleanHyperParam(default=True)]
        },

        'ElasticNet': {
            'l1_ratio': [[0.1, 0.3, 0.5, 0.7, 0.9],FloatHyperParam(min=0.001, max=0.1)],
            'selection': [['cyclic', 'random'],CategoricalHyperParam(choices=['cyclic', 'random'], default='cyclic')],
            'tol': [[.0000001, .000001, .00001, .0001, .001],FloatHyperParam(min=0.001, max=0.1)  ],
            'positive':[[True, False],BooleanHyperParam(default=True)]
        },

        'SGDRegressor': {
            'loss': [['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],CategoricalHyperParam(choices=['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], default='squared_loss')],
            'penalty': [['l2', 'l1', 'elasticnet'],CategoricalHyperParam(choices=['l2', 'l1', 'elasticnet'], default='l2')],
            'learning_rate': [['constant', 'optimal', 'invscaling'],CategoricalHyperParam(choices=['constant', 'optimal', 'invscaling'], default='constant')],
            'alpha': [[.0000001, .000001, .00001, .0001, .001],FloatHyperParam(min=0.001, max=0.1)]
        },
        'PassiveAggressiveRegressor': {
            'epsilon': [[0.01, 0.05, 0.1, 0.2, 0.5],FloatHyperParam(min=0.01, max=0.9)],
            'loss': [['epsilon_insensitive', 'squared_epsilon_insensitive'],CategoricalHyperParam(choices=['epsilon_insensitive', 'squared_epsilon_insensitive'], default='epsilon_insensitive')],
            'C': [[.0001, .001, .01, .1, 1, 10, 100, 1000],FloatHyperParam(min=0.001, max=1.0)],
        },
        'SGDClassifier': {
            'loss': [['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],CategoricalHyperParam(choices=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], default='hinge')],
            'penalty': [['l2', 'l1', 'elasticnet'],CategoricalHyperParam(choices=['l2', 'l1', 'elasticnet'], default='l2')],
            'learning_rate': [['constant', 'optimal', 'invscaling'],CategoricalHyperParam(choices=['constant', 'optimal', 'invscaling'], default='constant')],
            'alpha': [[.0000001, .000001, .00001, .0001, .001],FloatHyperParam(min=0.0000001, max=0.1)],
            'class_weight': [['balanced', None],CategoricalHyperParam(choices=['balanced', None], default='balanced')]
        },
        'PassiveAggressiveClassifier': {
            'loss': [['hinge', 'squared_hinge'],CategoricalHyperParam(choices=['hinge', 'squared_hinge'], default='hinge')],
            'class_weight': [['balanced', None],CategoricalHyperParam(choices=['balanced', None], default='balanced')],
            'C': [[0.01, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],FloatHyperParam(min=0.01, max=1.0)]
        },
        'LGBMClassifier': { 
            'boosting_type': [['gbdt', 'dart'],CategoricalHyperParam(choices=['gbdt', 'dart'], default='gbdt')]
            , 'min_child_samples': [[1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000], IntHyperParam(min=1, max=1000, default=100)]
            , 'num_leaves': [[2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250], IntHyperParam(min=2, max=1000, default=100)]
            , 'colsample_bytree': [[0.7, 0.9, 1.0],FloatHyperParam(min=0.2, max=1.0)]
            , 'subsample': [[0.7, 0.9, 1.0],FloatHyperParam(min=0.1, max=1.0)]
            , 'learning_rate': [[0.01, 0.05, 0.1],FloatHyperParam(min=0.01, max=0.5)]
            , 'n_estimators': [[5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000], IntHyperParam(min=5, max=2000, default=100)]

        },
        'LGBMRegressor': {
            'boosting_type': [['gbdt', 'dart'],CategoricalHyperParam(choices=['gbdt', 'dart'], default='gbdt')]
            , 'min_child_samples': [[1, 5, 7, 10, 15, 20, 35, 50, 100, 200, 500, 1000], IntHyperParam(min=1, max=1000, default=100,step=1)]
            , 'num_leaves': [[2, 4, 7, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80, 100, 125, 150, 200, 250], IntHyperParam(min=2, max=1001, default=100,step=1)]
            , 'colsample_bytree': [[0.7, 0.9, 1.0],FloatHyperParam(min=0.2, max=1.0)]
            , 'subsample': [[0.7, 0.9, 1.0],FloatHyperParam(min=0.1, max=1.0)]
            , 'learning_rate': [[0.01, 0.05, 0.1],FloatHyperParam(min=0.01, max=0.5)]
            , 'n_estimators': [[5, 20, 35, 50, 75, 100, 150, 200, 350, 500, 750, 1000], IntHyperParam(min=5, max=2000, default=100,step=1)]
        }

        , 'CatBoostClassifier': {
            'depth': [[1, 2, 3, 5, 7, 9, 12, 15], IntHyperParam(min=1, max=16, default=10)]
            , 'l2_leaf_reg': [[.0000001, .000001, .00001, .0001, .001, .01, .1],FloatHyperParam(min=0.000001, max=0.1)]
            , 'learning_rate': [[0.01, 0.05, 0.1, 0.15, 0.2, 0.3],FloatHyperParam(min=0.001, max=0.3)]
            , 'random_strength':[[1e-9,2,5, 10],FloatHyperParam(min=0.00001, max=10.0)]
            , 'bagging_temperature':[[0.0,0.2,0.4,0.6,0.7, 0.9, 1.0],FloatHyperParam(min=0.0, max=1.0)]
        }

        , 'CatBoostRegressor': {
            'depth': [[1, 2, 3, 5, 7, 9, 12, 15,16], IntHyperParam(min=1, max=16, default=1)]
            , 'l2_leaf_reg': [[.0000001, .000001, .00001, .0001, .001, .01, .1],FloatHyperParam(min=0.001, max=0.1)]
            , 'learning_rate': [[0.01, 0.05, 0.1, 0.15, 0.2, 0.3],FloatHyperParam(min=0.001, max=0.3)]
            , 'random_strength':[[1e-9,2,5, 10],FloatHyperParam(min=0.001, max=10.0)]
            , 'bagging_temperature':[[0.0,0.2,0.4,0.6,0.7, 0.9, 1.0],FloatHyperParam(min=0.0, max=1.0)]
        }

        , 'LinearSVR': {
            'C': [[0.5, 0.75, 0.85, 0.95, 1.0],FloatHyperParam(min=0.2, max=1.0)]
            , 'epsilon': [[0.0, 0.05, 0.1, 0.15, 0.2],FloatHyperParam(min=0.0, max=0.2)]
        }

        , 'LinearSVC': {
            'C': [[0.5, 0.75, 0.85, 0.95, 1.0],FloatHyperParam(min=0.2, max=1.0)]
        },

        'SVC':{
            'C': [[0.5, 0.75, 0.85, 0.95, 1.0],FloatHyperParam(min=0.2, max=1.0)]
            ,"kernel":[["rbf", "poly", "linear", "sigmoid"],CategoricalHyperParam(choices=["rbf", "poly", "linear", "sigmoid"], default='linear')]
        },

        "SVR":{
            'C': [[0.5, 0.75, 0.85, 0.95, 1.0],FloatHyperParam(min=0.2, max=1.0)]
            ,"kernel":[["rbf", "poly", "linear", "sigmoid"],CategoricalHyperParam(choices=["rbf", "poly", "linear", "sigmoid"], default='linear')]
        },

        'BernoulliNB':{
            "alpha":  [[0.0,0.1,0.4,0.6,0.8, 1.0],FloatHyperParam(min=0.0, max=1.0)],
            "binarize": [[0.0,0.2,0.5,0.7, 1.0,None],FloatHyperParam(min=0.0, max=1.0)],
            "fit_prior": [[True, False],BooleanHyperParam(default=True)]

        },

        'GaussianNB':{
            'var_smoothing':[[1e-09,1e-08,1e-05,1e-03,1e-01,0.1],FloatHyperParam(min=0.0000001, max=0.3)]
        }, 

        'MultinomialNB':{
            "alpha":  [[0.0, 1.0], IntHyperParam(min=0, max=1, default=1)],
            "fit_prior": [[True, False],BooleanHyperParam(default=True)]
        },

       

        'DecisionTreeClassifier':{
            "criterion": [["entropy", "gini"],CategoricalHyperParam(choices=["entropy", "gini"], default='gini')],
            'max_features': [['auto', 'sqrt', 'log2', None],CategoricalHyperParam(choices=['auto', 'sqrt', 'log2', None], default='auto')],
            'min_samples_split': [[2, 5, 20, 50, 100], IntHyperParam(min=2, max=200, default=20)],
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100], IntHyperParam(min=1, max=200, default=20)]
        },

        'KNeighborsClassifier': {
            "n_neighbors": [[2,4,6,8,10,12,14,16,18,20],IntHyperParam(min=2, max=30, default=10)  ],
            "weights": [["uniform", "distance"],CategoricalHyperParam(choices=["uniform", "distance"], default='uniform')],
            "algorithm": [["ball_tree", "kd_tree", "brute"],CategoricalHyperParam(choices=["ball_tree", "kd_tree", "brute"], default='brute')],
            "leaf_size":[[2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48,50],IntHyperParam(min=2, max=50, default=10)  ],
            "metric":  [["minkowski", "euclidean", "manhattan", "chebyshev"],CategoricalHyperParam(choices=["minkowski", "euclidean", "manhattan", "chebyshev"], default='euclidean')],
            "p":[[1,2,3],IntHyperParam(min=1, max=3, default=1)  ],
        },

         'DecisionTreeRegressor': {
            'max_features': [['auto', 'sqrt', 'log2', None],CategoricalHyperParam(choices=['auto', 'sqrt', 'log2', None], default='auto')],
            'min_samples_split': [[2, 5, 20, 50, 100], IntHyperParam(min=2, max=200, default=20,step=1)],#step=11
            'min_samples_leaf': [[1, 2, 5, 20, 50, 100], IntHyperParam(min=2, max=200, default=20,step=1)] #step=11
         },

        'KNeighborsRegressor': {
            "n_neighbors": [[2,4,6,8,10,12,14,16,18,20],IntHyperParam(min=2, max=30, default=10,step=1)  ],#step=4
            "weights": [["uniform", "distance"],CategoricalHyperParam(choices=["uniform", "distance"], default='uniform')],
            "algorithm": [["ball_tree", "kd_tree", "brute"],CategoricalHyperParam(choices=["ball_tree", "kd_tree", "brute"], default='brute')],
            "leaf_size":[[2,4,6,8,10,12,14,16,18,20,24,28,32,36,40,44,48,50],IntHyperParam(min=2, max=50, default=10,step=1)  ],#step=4
            "metric":  [["minkowski", "euclidean", "manhattan", "chebyshev"],CategoricalHyperParam(choices=["minkowski", "euclidean", "manhattan", "chebyshev"], default='euclidean')]

        },
        'KerasRegressor':{
             'dropout': [[0.0, 0.2, 0.4, 0.6, 0.8],FloatHyperParam(min=0.0, max=0.5)]
            , 'kernel_initializer': [['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],CategoricalHyperParam(choices=['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='uniform')]
            , 'activation': [['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'],CategoricalHyperParam(choices=['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'], default='relu')]
            , 'batch_size': [[16, 32, 64, 128, 256, 512],IntHyperParam(min=16, max=1000, default=250)]
            , 'epochs': [[2, 4, 6, 10, 20],IntHyperParam(min=2, max=50, default=10)  ]
            , 'optimizer': [['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],CategoricalHyperParam(choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'], default='Adam')]
            , "min_units": [[10,20,30,40,50,60], IntHyperParam(min=10, max=50, default=10) ]
            , "learn_rate" : [[0.001, 0.01, 0.1, 0.2, 0.3],FloatHyperParam(min=0.0001, max=0.3)]
            ,  'hidden_layers': [[(1,),(0.5,),(2,),(1, 1),(0.5, 0.5),(2, 2),(1, 1, 1),(1, 0.5, 0.5),(0.5, 1, 1),(1, 0.5, 0.25),(1, 2, 1),(1, 1, 1, 1),(1, 0.66, 0.33, 0.1),(1, 2, 2, 1) ],CategoricalHyperParam(choices=[(1,),(0.5,),(2,),(1, 1),(0.5, 0.5),(2, 2),(1, 1, 1),(1,0.75,0.5),(1, 0.5, 0.5),(1,0.75,0.5),(0.5, 1, 1),(1, 0.5, 0.25),(1, 2, 1),(1, 1, 1, 1),(1, 0.66, 0.33, 0.1),(1, 2, 2, 1) ], default=(1,0.75,0.5))]
            , "final_activation" : [['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'],CategoricalHyperParam(choices=['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'], default='relu')]
        },
        'KerasClassifier':{
           'dropout': [[0.0, 0.2, 0.4, 0.6, 0.8],FloatHyperParam(min=0.0, max=0.5)]
            , 'kernel_initializer': [['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],CategoricalHyperParam(choices=['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'], default='uniform')]
            , 'activation': [['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'],CategoricalHyperParam(choices=['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'], default='relu')]
            , 'batch_size': [[16, 32, 64, 128, 256, 512],IntHyperParam(min=16, max=1000, default=250)]
            , 'epochs': [[2, 4, 6, 10, 20],IntHyperParam(min=2, max=50, default=10)  ]
            , 'optimizer': [['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],CategoricalHyperParam(choices=['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'], default='Adam')]
            , "min_units": [[10,20,30,40,50,60], IntHyperParam(min=10, max=50, default=10) ]
            , "learn_rate" : [[0.001, 0.01, 0.1, 0.2, 0.3],FloatHyperParam(min=0.0001, max=0.3)]
            ,  'hidden_layers': [[(1,),(0.5,),(2,),(1, 1),(0.5, 0.5),(2, 2),(1, 1, 1),(1, 0.5, 0.5),(1,0.75,0.5),(0.5, 1, 1),(1, 0.5, 0.25),(1, 2, 1),(1, 1, 1, 1),(1, 0.66, 0.33, 0.1),(1, 2, 2, 1) ],CategoricalHyperParam(choices=[(1,),(0.5,),(2,),(1, 1),(0.5, 0.5),(1,0.75,0.5),(2, 2),(1, 1, 1),(1, 0.5, 0.5),(0.5, 1, 1),(1, 0.5, 0.25),(1, 2, 1),(1, 1, 1, 1),(1, 0.66, 0.33, 0.1),(1, 2, 2, 1) ], default=(1,0.75,0.5))]
            , "final_activation" : [['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'],CategoricalHyperParam(choices=['tanh', 'softmax', 'elu', 'softplus', 'softsign', 'relu', 'sigmoid', 'linear'], default='relu')]
      }
       
        }

    def __init__(self):
        super().__init__()
        pass

    @classmethod
    def get_model_by_name(cls,model_name=None):
        if model_name in cls.Classifiers:
            return  cls.Classifiers[model_name]
        elif model_name in cls.Regressors:
            return  cls.Regressors[model_name]
        else :
            raise ValueError("{} is not a Valid Model Name".format(model_name))
        
    @classmethod
    def set_params(cls,model=None,params=None):
        for k, v in params.items():
            model = model.set_params(**{k: v})
        return model.set_params(**params)

    @classmethod
    def get_params(cls,model=None):
        if type(model)in  [CatBoostRegressor,CatBoostClassifier]:
            return model.get_all_params()
        else:
            return model.get_params()

    @classmethod
    def get_classifiers(cls):
        return cls.Classifiers

    @classmethod
    def get_regressors(cls):
        return cls.Regressors
    
    @classmethod
    def get_search_space(cls,model_name):
        space=cls.space[model_name]
        search_space={}
        for k,v in space.items():
            search_space[k]=v[0]
            
        return search_space

    @classmethod
    def get_btb_space(cls,model_name):
        space=cls.space[model_name]
        btb_space={}
        for k,v in space.items():
            btb_space[k]=v[1]
            
        return btb_space




