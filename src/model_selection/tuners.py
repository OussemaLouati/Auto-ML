from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from .model import Model 
import copy
import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Tuner: 

    def __init__(self):
        super().__init__()

    @classmethod
    def tune(cls, X_train=None, y_train=None ,estimator=None, param=None, scoring=None, cv=5, n_iter=10, tuner='random',n_jobs=-1):
        if tuner == 'random': 
            rand= RandomizedSearchCV(estimator=estimator.get_estimator(), param_distributions=param, n_iter=n_iter, scoring=scoring, cv=cv, n_jobs=n_jobs)
            result=rand.fit(X_train, y_train)
        elif tuner=='grid':
            grid = GridSearchCV(estimator=estimator.get_estimator(), param_grid = param,scoring=scoring, cv=cv, n_jobs=n_jobs)
            result=grid.fit(X_train, y_train)
        else :
            raise ValueError("{} is not a valid Tuner, choose either 'random or 'grid ".format(tuner)) 
        return Model(problem_type=estimator.problem_type, algorithm=estimator.algorithm, hyperparameters=result.best_params_,num_cols=len(list(X_train.columns)), metric= estimator.metric)



