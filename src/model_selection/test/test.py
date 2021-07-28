
from importlib import import_module
from model_selection.model import Model
from model_selection.method import Algorithm
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
import sys
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn import linear_model
from xgboost import XGBRegressor
from model_selection.selection import Selection
from model_selection.tuners import Tuner

#read data
currentPath = os.path.dirname(sys.argv[0]) + "/"
Tunisair_df=pd.read_csv(currentPath+'Tunisair.csv', engine='python')
Tunisair_df.index=range(Tunisair_df.shape[0])
Tunisair_df.drop(columns=['STA'],inplace=True)
Tunisair_df.drop(columns=['STD'],inplace=True)
Tunisair_df.drop(columns=['DATOP'],inplace=True)
X=Tunisair_df.drop(columns=['Target'])
y=Tunisair_df['Target']

X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)



#Initialize different generic models using their names you can consult "./method.py" to see the algorithms that are already prepared for this library,
# OR pass a user provided model in fact any model passed as 'cus_model' (custom model) will be converted to a sckit learn estimator
#we will start by creating a kerasregressor estimator, NOTE here that when using a keras model you can run """" tensorboard --logdir logs/ """"
#in terminal to Visualize training results through tensorboard
print("__________________________Model 1_______________________\n")
first_model=Model(problem_type="regression",algorithm="KerasRegressor",metric="MAE", num_cols=5)
first_model.fit(X_train,Y_train)
y_pred1=first_model.predict(X_test)
print(first_model.score(y_test,y_pred1))
print("____________________User provided Model________________\n")
xgb=XGBRegressor()
second_model=Model(problem_type="regression",cus_model=xgb ,metric="MAE")
second_model.fit(X_train,Y_train)
y_pred2=second_model.predict(X_test)
print(second_model.score(y_test,y_pred2))
print("_______________ Compatibility with scikit learn Pipeline_____________\n")
print("_______first model_______\n")

pip1 = Pipeline(steps=[('Regressor1',first_model)])
y_pre1=pip1.predict(X_test)
print(pip1.score(y_pre1, y_test))
print("\n_______second model_______\n")

pip2 = Pipeline(steps=[('Regressor2',second_model)])
y_pre2=pip2.predict(X_test)
print(pip2.score(y_pre2, y_test))

print("\n_______________ Get any type of error you want (Explained Variance for example) _____________\n\n")
''' You can see all the metrics used in the './metrics.py' '''
print('explained variance : {}'.format(first_model.error(y_test,y_pred1,metric="EV")))

print("\n_______________ Perform StratifiedKFold fit validate _____________\n\n")
errors, predictions,cv_error=second_model.kfold_fit_validate(X_train,Y_train, n_splits=5)
print("errors : {}".format(errors))
print("CV error : {}".format(cv_error))
print("predictions: {}".format(predictions))

print("\n_______________ Get a full a full and final report _____________\n\n")

first_model.final_report(y_test,y_pred1,chuncks=5)

print("\n_______________ Tuning a model _____________\n\n")
''' YOU should NOTE that estimator should be of Type 'Model' 
           tuners are : random or grid search 
           so let's tune the kerasRegressor model               '''
#get search space of keras regressor 
space=Algorithm.get_search_space("KerasRegressor")
#This will return also an estimator of type 'Model' initialized with the new tuned parameters
'''
tuned_model=Tuner.tune(X_train, Y_train ,estimator=second_model, param=space, cv=5, n_iter=10, tuner='random',n_jobs=-1)
tuned_model.fit(X_train,Y_train)
ypred_tuned=tuned_model.predict(X_test)
print(tuned_model.score(y_test,ypred_tuned)) '''


print("\n_______________ Select best model from a given list of Models _____________\n\n")
# if we don't specify models , models will initialized with all the regression models we have (for regression problems ) and same for classification
# models can be a mix of user defined or names of models
Selector=Selection(models=["Lasso","LinearRegression","LGBMRegressor"],judgment_metric="MAE",problem_type="regression")
#tune is a list with argumets (no_tuning= 0) , (random = 1) and (grid = 2) , writing 'random' or 1 is the same if len(tune)<len(models)
# difference (len(tune)-len(models)) number of models will be considered as no_tuning and won't be tuned
#eval set should have the following format [(X_train,Y_train),(X_test, y_test)], this return the best model
Selector.BestModelK(eval_set=[(X_train,Y_train),(X_test, y_test)], tune=[2,'grid',1],cv=2, n_iter=2)
be=Selector.best_estimator()
bp=Selector.best_params()

print("\n_______________ print results (report) _____________\n\n")
Selector.print_results()


#u can also get any model from the specified to the selection method
mdl=Selector.get_model(name="Lasso + Grid search")

print("\n_______________ Get matrix of models and their scores given certain metrics _____________\n\n")
#if u don't specify models or metrics you will just have all models and all metrics scores
Selector.get_model_V_error()
print('_______________________________________________')
print('_______')

Selector.get_model_V_error(models=["Lasso","LinearRegression"],metrics=["RMSE","EV","R2"])







