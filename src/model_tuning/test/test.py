
from importlib import import_module
from sklearn.model_selection import train_test_split
import pandas as pd 
import os
import sys,random
import numpy as np
from xgboost import XGBRegressor
from model_selection.tuners import Tuner
from model_tuning.optuna_tuner import OptunaTuner
from model_tuning.tuner import Raytuner
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

xgb=XGBRegressor()
op = OptunaTuner(model=xgb,n_trials=2,ASHA=True,problem_type="regression",judgment_metric="MAE")
df,model = op.optimize(X_train, X_test, Y_train, y_test)
print(df)

param = {
        # Perturb factor1 by scaling it by 0.8 or 1.2. Resampling
        # resets it to a value sampled from the lambda function.
        "factor_1": lambda: random.uniform(0.0, 20.0),
        # Perturb factor2 by changing it to an adjacent value, e.g.
        # 10 -> 1 or 10 -> 100. Resampling will choose at random.
        "factor_2": [1, 10, 100, 1000, 10000],
    }

raytuners = Raytuner(model)
analysis = raytuners.PopulationBasedTraining(10,param)
raytuners.get_best_param()