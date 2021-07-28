import sklearn
import xgboost
import catboost
import lightgbm
import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from importlib import import_module
from tabulate import tabulate
from .method import Algorithm 
from copy import copy
from sklearn.metrics import make_scorer
from .metrics import RegressionScorer, BinaryClassifierScorer, MultiClassClassifierScorer
from .constants import REGRESSION_METRICS , CLASSIFICATION_METRICS
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from .rc_deep_models import KerasModels
from .utils import Utils
import shap
import statistics
import tensorflow as tf
import keras
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class Model(BaseEstimator, ClassifierMixin):
    """Parameters:
    ------

            • problem_type (str) - "regression", "binary_classification" or "multiclass_classification"     

            • algorithm (str) - Machine learning algorithm to use

            • cus_model (ML Model) - Initialize this module with another model

            • metric (str) - Metric to use for the error
            
            • hyperparameters (dict) - Dictionary of hyperparameters.
            
            • num_cols (int) - Number of columns: this is used for the input layer of the keras Model
    """
 
    def __init__(self, problem_type=None, algorithm=None,cus_model=None,metric=None, hyperparameters={}, verbose=False,num_cols=None):
        self.problem_type = problem_type
        self.metric=metric
        if self.metric==None and problem_type=="regression":
            self.metric="RMSE"
        if self.metric==None and problem_type.lower()=='binary_classification':
            self.metric="ROC_AUC"
        if self.metric==None and problem_type.lower()=='multiclass_classification':
            self.metric="ACCURACY"
        if callable(cus_model):
            self.cus_model=Utils._import(str(cus_model.__class__))()
        else:
            self.cus_model=cus_model
        self.algorithm = algorithm if cus_model==None else cus_model.__class__.__name__
        self.hyperparameters = hyperparameters
        self.num_cols=num_cols 
        self.model_ = self._instantiate(self.cus_model) 
        self.verbose = verbose
        self.fittedModel_=None
        self.fitted_ = False 
        self.X_train=None
        self.Y_train=None
        self.X_test=None
        self.predicted=False
        self.cv_error = None
        self.cv_preds = None
        if problem_type=="regression":
            self.metric_dict=REGRESSION_METRICS
        else :
            self.metric_dict=CLASSIFICATION_METRICS   
        self.provided_params=hyperparameters.copy()
        #if self.algorithm[:8]!= 'CatBoost':
        self.set_params(hyperparameters=self.model_.get_params())
        log_dir = "logs\\" +self.algorithm+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        

    def _instantiate(self,cus_model=None):
        if self.algorithm in ["KerasClassifier","KerasRegressor"] :
            keras_model=Utils._import("keras.wrappers.scikit_learn."+self.algorithm)
            return keras_model(build_fn=KerasModels.build, 
                            hidden_layers=self.hyperparameters['hidden_layers'] if 'hidden_layers' in self.hyperparameters.keys() else (1, 0.75, 0.25),
                            num_cols =self.num_cols,
                            optimizer=self.hyperparameters['optimizer'] if 'optimizer' in self.hyperparameters.keys() else  'Adadelta',
                            dropout=self.hyperparameters['dropout'] if 'dropout' in self.hyperparameters.keys() else  0.2, 
                            kernel_initializer=self.hyperparameters['kernel_initializer'] if 'kernel_initializer' in self.hyperparameters.keys() else 'normal',
                            activation=self.hyperparameters['activation'] if 'activation' in self.hyperparameters.keys() else 'elu',
                            final_activation=self.hyperparameters['final_activation'] if 'final_activation' in self.hyperparameters.keys() else 'sigmoid',
                            min_units=self.hyperparameters['min_units'] if 'min_units' in self.hyperparameters.keys() else 10 ,
                            problem_type=self.problem_type,
                            batch_size=self.hyperparameters['batch_size'] if 'batch_size' in self.hyperparameters.keys() else 250,
                            epochs=self.hyperparameters['epochs'] if 'epochs' in self.hyperparameters.keys() else 30,
                            learn_rate=self.hyperparameters['learn_rate'] if 'learn_rate' in self.hyperparameters.keys() else 0.01,
                            metric=self.metric,
                            verbose=0)
        
        mdl=Algorithm.get_model_by_name(self.algorithm)() if cus_model==None else cus_model
        if self.hyperparameters!={}:
            mdl=Algorithm.set_params(mdl,self.hyperparameters)
        return mdl 

    def fit(self, X, y,cat_features=[],plot=False,**args):
        if (((self.algorithm in Algorithm.get_classifiers()) and (self.problem_type.lower() not in ['multiclass_classification','multiclass_classification']))
           or ((self.algorithm in Algorithm.get_regressors()) and (self.problem_type.lower() != 'regression'))):
            raise ValueError("Theres is no correspondence between the Algorithm you chose and the problem type")
        x_train=X
        y_train=y
        
        if((type(x_train) != pd.SparseDataFrame) and
           (type(x_train) != pd.DataFrame)):
            raise ValueError("x_train must be a DataFrame")
        if (type(y_train) != pd.core.series.Series):
            raise ValueError("y_train must be a Series")
        if self.algorithm[0:8]=="CatBoost":
            indices=Utils.get_columns_indices(list(x_train.columns),cat_features)
            self.fittedModel_=self.model_.fit(x_train,y_train,cat_features=indices,plot=plot,verbose=False)
            #self.set_params(hyperparameters=self.fittedModel_.get_all_params())
        elif self.algorithm[0:4]=="LGBM":
            self.fittedModel_=self.model_.fit(x_train,y_train,categorical_feature=cat_features,verbose = False)
        elif self.algorithm[0:3]=="XGB":
            self.fittedModel_= self.model_.fit(x_train,y_train, verbose=False)
        elif self.algorithm in ["KerasClassifier","KerasRegressor"]:  
            self.model_.fit(x_train,y_train,callbacks=[self.tensorboard_callback],verbose=0)  
        else :    
            self.fittedModel_=self.model_.fit(x_train,y_train)
        self.fitted_ = True
        self.X_train=X
        self.y_train=y
        if self.verbose:
            print("\n{} : with provided params : {} complete.".format(self.algorithm, self.hyperparameters))
        return self

    def predict(self, X,y=None):
        x_test=X
        self.X_test=X
        self.predicted=True
        if self.fitted_ and type(self.model_) in [KerasRegressor, KerasClassifier]:  
            return self.model_.predict(x_test)
        if not callable(getattr(self.fittedModel_, "predict")):
            raise ValueError("predict is not callable")
        if ((type(x_test) != pd.SparseDataFrame) & (type(x_test) != pd.DataFrame)):
            raise ValueError("x_test must be a DataFrame")
        if self.fitted_:
            return self.fittedModel_.predict(x_test)
        else: 
            print ("You must Perform fit before trying to predict")

    def set_params(self,**params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def get_params(self,deep=True):
        return {"problem_type" :self.problem_type,  
                "algorithm" : self.algorithm ,
                "hyperparameters" : self.hyperparameters, 
                "cus_model":self.cus_model,
                "verbose": self.verbose,
                "metric":     self.metric,
                "num_cols":self.num_cols 
                    }

    def get_estimator(self):
        if self.fitted_:
            if self.algorithm in ["KerasClassifier","KerasRegressor"]:  
                return copy(self.model_)
            else:
                return copy(self.fittedModel_)
        else :
            return copy(self.model_)

    
    def score(self, X=None, y=None,):
    
        if self.problem_type=="regression":
            error="RMSE"  if self.metric== None else self.metric
            errors=RegressionScorer.error(X, y)
        elif self.problem_type=="binary_classification":
            error="ROC_AUC" if self.metric== None else self.metric
            errors=BinaryClassifierScorer.error(X, y)
        elif self.problem_type=="multiclass_classification" :
            error="ACCURACY" if self.metric== None else self.metric
            errors=MultiClassClassifierScorer.error(X, y)
        else: 
            raise ValueError ('Not a Valid Problem Type : SO NO ERRORS! NO METRICS')
        return errors[error]
        
    def error(self, X=None, y=None,metric=None):
        y_true=X
        y_predicted=y
        if self.problem_type=="regression":
            errors=RegressionScorer.error(y_true, y_predicted)
        elif self.problem_type=="binary_classification":
            errors=BinaryClassifierScorer.error(y_true, y_predicted)
        elif self.problem_type=="multiclass_classification" :
            errors=MultiClassClassifierScorer.error(y_true, y_predicted)
        else: 
            raise ValueError ('Not a Valid Problem Type : SO NO ERRORS! NO METRICS')
        if metric==None :
            
            return errors
        else:
            return errors[metric]

    def final_report(self,X=None, y=None,chuncks=4):
        errors=self.error(X,y)
        neg=["NLL","MAE","MSE","MSLE","MeAE","RMSE","RMSLE"]
        print("\n")
        print("Algorithm used : {}".format(self.algorithm))
        print('_______')
        print("Problem Type : {}".format(self.problem_type))
        print('_______')
        print("Metric used as default score : {}".format(self.metric))
        print('_______')
        print("Hyperparameters provided : {}".format(self.provided_params))
        print('_______')
        print("Hyperparameters used : {}".format(self.hyperparameters))
        print('_______')
        print("Errors : ")
        print("_____________________________________________________________\n")
        table = [ [k,v]  if (k!=self.metric and k not in neg) 
        else [k+' (Default scorer)', v] if k==self.metric and k not in neg 
        else [k, str(v)+' **'] if k in neg and  k!=self.metric else [k+' (Default scorer)', str(v)+' **']  for k, v in errors.items() ]
        headers = ["Metrics", "Scores"]
        print(tabulate(table, headers, tablefmt="grid"))
        print("** For these metrics we are using their negative values")
        if self.problem_type=="regression":
            print("_____________________________________________________________\n")
            RegressionScorer.report(X,y, chuncks=chuncks)


    def kfold_fit_validate(self, X, y, n_splits):
        y_pred = np.empty(y.shape)
        list_errors = [0]*n_splits
        kf = StratifiedKFold(n_splits, shuffle=True, random_state=None)
        i=0
        for train_idx, test_idx in kf.split(X, y):
            x_tr = X.iloc[train_idx, :]
            y_tr = y.iloc[train_idx]
            x_val = X.iloc[test_idx, :]
            y_val = y.iloc[test_idx]
            if len(np.unique(y_tr)) > 1:
                self.model_.fit(x_tr, y_tr)
                y_pred[test_idx] = self.model_.predict(x_val)
            else:
                y_predicted[test_idx] = y_tr[0]
            list_errors[i] = self.score(y_val, y_pred[test_idx])
            i=i+1
        self.cv_error =  statistics.mean(list_errors)
        self.cv_preds = y_pred
        if self.verbose:
            print("StratifiedKFold with {} splits on {} complete.".format(n_splits,self.algorithm))
        return list_errors, y_pred,self.cv_error

    def load_model(self,filepath):
        import json
        with open(filepath, 'r') as file:
            dict_ = json.load(file)

        self.problem_type = dict_['problem_type']
        self.algorithm = dict_['algorithm']
        self.hyperparameters = dict_['hyperparameters']
        self.cus_model=dict_['cus_model'] 
        self.metric=dict_['metric']
        self.num_cols=dict_['num_cols'] 
        self.X_train = np.asarray(dict_['X_train']) if dict_['X_train'] != 'None' else None
        self.Y_train = np.asarray(dict_['Y_train']) if dict_['Y_train'] != 'None' else None

    def save_model(self,filepath):
        import json
        dict_ = dict()
        dict_['problem_type'] = self.problem_type
        dict_['algorithm'] = self.algorithm
        dict_['hyperparameters'] = self.hyperparameters
        dict_['cus_model'] = self.problem_type
        dict_['metric'] = self.metric
        dict_['num_cols'] = self.num_cols
        dict_['X_train'] = self.X_train.tolist() if self.X_train is not None else 'None'
        dict_['Y_train'] = self.Y_train.tolist() if self.Y_train is not None else 'None'
        json_txt = json.dumps(dict_, indent=4)
        with open(filepath, 'w') as file:
            file.write(json_txt)

    def _prepare_shap(self):
        shap.initjs()
        model=self.get_estimator()
        model.fit(self.X_train,self.y_train)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(self.X_train)
        return explainer,shap_values


    def explain_model(self,n_pred=0,plot_type=None,n_observations=500):
        explainer,shap_values=self._prepare_shap()
        s='st'
        if n_pred==1:
            s="nd"
        elif n_pred==2:
            s="rd"
        elif n_pred>2:
            s="th"
        else :
            pass
        print('\nVisualize the {}\'{} prediction\'s explanation'.format(n_pred+1,s))
        print('\n______________________________________________')
        shap.force_plot(explainer.expected_value, shap_values[0,:], self.X_train.iloc[n_pred,:],matplotlib=True)
        print("\nFeature Importance")
        print('\n______________________________________________')
        explainer,shap_values=self._prepare_shap()
        shap.summary_plot(shap_values, self.X_train,plot_type=plot_type)
        print("\nDecision Plot")
        print('\n______________________________________________')
        explainer,shap_values=self._prepare_shap()
        features_display=self.X_train.loc[np.random.permutation(self.X_train.index)[:n_observations]]
        shap.decision_plot(explainer.expected_value, shap_values,features_display,ignore_warnings=True)

    def dependence_plot(self,column=None):
        explainer,shap_values=self._prepare_shap()
        shap.dependence_plot(column, shap_values, self.X_train)
        
    