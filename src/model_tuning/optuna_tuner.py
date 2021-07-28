import optuna
import pandas as pd
import os
import functools
from model_selection.method import Algorithm
from model_selection.model import Model
from optuna.pruners import SuccessiveHalvingPruner,MedianPruner

class OptunaTuner:
    
    def __init__(self,model=None,space=None,n_trials=100,ASHA=False,median_pruner=False,n_warmup_steps=4,problem_type=None,judgment_metric=None):
        self.judgment_metric=judgment_metric
        if judgment_metric==None and problem_type=="regression":
            self.judgment_metric="RMSE"
        if judgment_metric==None and problem_type.lower()=='binary_classification':
            self.judgment_metric="ROC_AUC"
        if judgment_metric==None and problem_type.lower()=='multiclass_classification':
            self.judgment_metric="ACCURACY"  
        if type(model)== str:
            self.model=Model(problem_type=problem_type, algorithm=model,metric=self.judgment_metric)
        elif type(model)== Model:
            self.model=model
        else:
            self.model=Model(problem_type=problem_type, cus_model=model,metric=self.judgment_metric) 
        if space is None : 
            try:
                self.space=Algorithm.get_search_space(self.model.algorithm)
            except:
                raise ValueError('You should provide a search space')
        else : 
            self.space=space 
        self.ASHA=ASHA
        self.median_pruner=median_pruner
        self.n_warmup_steps=n_warmup_steps
        self.n_trials=n_trials
    
    def optimize(self,X_train,X_test,Y_train, y_test):
        if self.ASHA:
            study = optuna.create_study(direction='maximize',pruner=SuccessiveHalvingPruner())
        elif self.median_pruner:
            study = optuna.create_study(direction='maximize',pruner=MedianPruner(n_warmup_steps=self.n_warmup_steps))
        else :
            study = optuna.create_study(direction='maximize')
        study.optimize(functools.partial(self.obj, X_train,Y_train,X_test, y_test), n_trials=self.n_trials)
        trial = study.best_trial
        print(self.model.metric+': {}'.format(trial.value))
        print("Best hyperparameters: {}".format(trial.params))
        tuned_model=Model(cus_model=self.model.get_estimator(),problem_type=self.model.problem_type,metric=self.model.metric, hyperparameters=trial.params)
        df = study.trials_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] == self.n_trials
        return df,tuned_model

    def obj(self,X_train,Y_train,X_test, y_test,trial):
        space_=self._search_space_from_dict(self.space,trial)
        mdl_tuna=Model(cus_model=self.model.get_estimator(),problem_type=self.model.problem_type,metric=self.model.metric, hyperparameters=space_)
        mdl_tuna.fit(X_train, Y_train)
        y_pred = mdl_tuna.predict(X_test)
        return mdl_tuna.score(y_test, y_pred)
        
    def _search_space_from_dict(self,dict_hyperparams,trial):
        hyperparams = {}
        if not isinstance(dict_hyperparams, dict):
            raise TypeError('Hyperparams must be a dictionary.')
        for name, hyperparam in dict_hyperparams.items():
            hp_type = type(hyperparam[0])
            if hp_type == int:
                hyperparams[name]= trial.suggest_int(name, hyperparam[0], hyperparam[-1])
            if hp_type == float:
                hyperparams[name]= trial.suggest_loguniform(name, hyperparam[0], hyperparam[-1])
            elif hp_type == bool or hp_type == str:
                hyperparams[name]= trial.suggest_categorical(name, hyperparam)
        return hyperparams