from .model import Model 
from .method import Algorithm
from .tuners import Tuner
from .utils import Utils 
from .btb_utilities import BtbUtils
from tqdm import trange,tqdm
from time import sleep
import sys
from .constants import TUNERS, CLASSIFICATION_METRICS, REGRESSION_METRICS ,METRIC_CLASSES
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.metrics import  make_scorer
from .utils import Utils
from tabulate import tabulate
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

class Selection:
    search={
         "random":"Randomized search",
         "grid" : "Grid search",
         "gp": " GPTuner" , 
         "gp_ei": "GPEiTuner" , 
         "uniform" : "UniformTuner" 
           }
    def __init__(self,models_config=None,judgment_metric=None,problem_type=None,tune_all=0):
        super().__init__()
        self.tune_all=tune_all
        self.models_config=models_config
        if self.models_config==None:
            if  problem_type=="regression" : 
                d=Algorithm.get_regressors()
            if  problem_type in ['multiclass_classification','binary_classification'] :
                d=Algorithm.get_classifiers()
            for e in list(d.keys()) :
                tune=tune_all if type(tune_all)==str else TUNERS[tune_all]
                assert(tune in ['no_tuning','random','grid',"gp","gp_ei","uniform"]), 'not a valid tuning method'
                self.models_config[e]['tune']=tune
        self.models=list(self.models_config.keys())
        self.judgment_metric=judgment_metric
        self.problem_type=problem_type
        self.best_model=None
        self.best_param=None
        self.fitted=False
        self.scores=None
        if judgment_metric==None and problem_type=="regression":
            self.judgment_metric="RMSE"
        if judgment_metric==None and problem_type.lower()=='binary_classification':
            self.judgment_metric="ROC_AUC"
        if judgment_metric==None and problem_type.lower()=='multiclass_classification':
            self.judgment_metric="ACCURACY"  
        if problem_type=="regression":
            self.metric_dict=REGRESSION_METRICS
        else :
            self.metric_dict=CLASSIFICATION_METRICS
        self.scoring=self._make_scorer(self.judgment_metric)
        self.all_errors={}
        self.Bmodel_name=None
        self.bestscore=None

    def _make_scorer(self,metric):
        clas=METRIC_CLASSES[metric]
        temp=clas.split('_')
        c=Utils._import("sklearn.metrics."+clas)
        if temp[-1].lower() in ["macro","micro","weighted"]:
            return make_scorer(c, average=temp[-1].lower())
        else : 
            return make_scorer(c)

    def BestModelK(self, eval_set=None, selection_type='brute_force', btb_n_iter=100, verbose=False, model_selector='ucb1'): 
        try:
            X_train=eval_set[0][0]
            Y_train=eval_set[0][1]
            X_val=eval_set[1][0]
            Y_val=eval_set[1][1]
        except:
            print('Eval_set should have the following format : [ ( X_train , y_train ) , ( x_val , y_val ) ]')
        n_cols=len(list(X_train.columns))
        models_det={}
        if (selection_type=='brute_force'):
            scores=Utils.list_to_dict(self.models)
            c=tqdm(self.models, leave=True)
            int_dict={}
            count=0
            for model in c:
                config=self.models_config[model]

                if type(model)== str:
                    mdl=Model(problem_type=self.problem_type, algorithm=model,metric=self.judgment_metric, num_cols=n_cols)
                else:
                    mdl=Model(problem_type=self.problem_type, cus_model=model,metric=self.judgment_metric,num_cols=n_cols)
                cnt=Utils.get_count(int_dict,mdl.algorithm)
                op='' if cnt==1 else '_'+str(cnt)
                desc=model+op if type(model)==str else model.__class__.__name__+op 
                int_dict[desc]=0
                c.set_description(desc)
                c.refresh() 
                sleep(1.0)
                mdl.fit(X_train, Y_train)
                pred=mdl.predict(X_val)
                error=mdl.score(Y_val,pred)
                self.all_errors[desc]=mdl.error(Y_val,pred)
                tune_= config["tuner"] if "tuner" in config else "no_tuning"
                tune=tune_ if type(tune_)==str else TUNERS[tune_]
                assert(tune in ['no_tuning','random','grid',"gp","gp_ei","uniform"]), 'not a valid tuning method'
                models_det[desc]=mdl
                scores[desc]=error
                if tune in ['random','grid']:
                    new_desc='{} Tuning on {}...'.format(self.search[tune],mdl.algorithm)
                    c.set_description(new_desc)
                    c.refresh() 
                    sleep(1.0)
                    space=Algorithm.get_search_space(mdl.algorithm) if "space" not in config else config["space"]
                    n_iter= 100 if "n_iter" not in config else config["n_iter"]
                    cv= 5 if "cv" not in config else config["cv"]
                    n_jobs=-1 if "n_jobs" not in config else config["n_jobs"]
                    mdl1=Tuner.tune(X_train=X_train, y_train=Y_train , estimator=mdl, scoring=self.scoring, param=space, tuner=tune,cv=cv, n_iter=n_iter ,n_jobs=1 if mdl.algorithm[:5]=="Keras" else n_jobs)
                    mdl1.fit(X_train, Y_train)
                    pred1=mdl1.predict(X_val)
                    error1=mdl1.score(Y_val,pred1)
                    sleep(1.0)
                    t=desc+' + {}'.format(self.search[tune])
                    models_det[t]=mdl1
                    scores[t]=error1
                    self.all_errors[t]=mdl1.error(Y_val,pred1)
                
                if tune in ["gp","gp_ei","uniform"]:
                    new_desc='{} on {}...'.format(self.search[tune],mdl.algorithm)
                    c.set_description(new_desc)
                    c.refresh() 
                    sleep(1.0)
                    space=Algorithm.get_search_space(mdl.algorithm) if "space" not in config else BtbUtils.to_btb_space(config["space"])
                    n_iter= 100 if "n_iter" not in config else config["n_iter"]
                    space=Algorithm.get_btb_space(mdl.algorithm)
                    space_tunable=BtbUtils.as_tunable(space)
                    tuners=BtbUtils.load_tuner(tune)
                    tuner=tuners(space_tunable)
                    best_score = -9999
                    best_params=None
                    for i in range(n_iter),:
                        proposal = tuner.propose()
                        mdl1 = Model(problem_type=self.problem_type, cus_model=mdl.get_estimator(), metric=self.judgment_metric, num_cols=n_cols, hyperparameters=proposal)
                        #model = candidates[candidate](**parameters)
                        mdl1.fit(X_train, Y_train)
                        y_pred=mdl1.predict(X_val)
                        score = mdl.score(Y_val,y_pred)
                        if score > best_score:
                            best_params = proposal
                            best_score = score
                        tuner.record(proposal, score)
                    mdl1 = Model(problem_type=self.problem_type, cus_model=mdl.get_estimator(), metric=self.judgment_metric, num_cols=n_cols, hyperparameters=best_params)  
                    mdl1.fit(X_train, Y_train)
                    pred1=mdl1.predict(X_val)
                    error1=mdl1.score(Y_val,pred1)
                    sleep(1.0)
                    t=desc+' + {}'.format(self.search[tune])
                    models_det[t]=mdl1
                    scores[t]=best_score
                    self.all_errors[t]=mdl1.error(Y_val,pred1)
                    
                count=count+1
                
            bestscore=-9999 
            bestmodel=''
            for name,score in scores.items():
                if score > bestscore:
                    bestmodel=name
                    bestscore=score

        
        elif (selection_type=='btb'):
            if model_selector not in ['uniform','ucb1','bestk','bestkvel','purebestkvel','recentk']:
                raise ValueError('{} is not a valid Selector.'.format(model_selector))
            count=0            
            candidates=BtbUtils.prepare_candidates(self.models)
            tuners={}
            for c in candidates.keys() :
                config=self.models_config[c]
                if type(c)== str:
                    mdl=Model(problem_type=self.problem_type, algorithm=c,metric=self.judgment_metric, num_cols=n_cols)
                else:
                    mdl=Model(problem_type=self.problem_type, cus_model=c,metric=self.judgment_metric,num_cols=n_cols) 
                space=Algorithm.get_btb_space(mdl.algorithm) if "space" not in config else BtbUtils.to_btb_space(config["space"])
                space_tunable=BtbUtils.as_tunable(space)
                tune_= config["tuner"] if "tuner" in config else "gp"
                tune=tune_ if type(tune_)==str else TUNERS[tune_]
                assert(tune in ['uniform','gp','gp_ei']), 'not a valid tuning method'
                tuner=BtbUtils.load_tuner(tune)
                tuners[c]=tuner(space_tunable)
                count=count+1
            select_type=BtbUtils.load_selector(model_selector)
            selector=select_type(list(candidates.keys()))
            c=tqdm(range(btb_n_iter),leave=True)
            bestscore = -9999
            scores={}
            d={}
            for i in c:  
                for e in candidates.keys():
                    d[e]=tuners[e].scores
                candidate = selector.select(d)
                parameters = tuners[candidate].propose()
                model = Model(problem_type=self.problem_type, algorithm=candidate, metric=self.judgment_metric, num_cols=n_cols, hyperparameters=parameters)
                #model = candidates[candidate](**parameters)
                model.fit(X_train, Y_train)
                y_pred=model.predict(X_val)
                er=model.error(Y_val,y_pred)
                score = model.score(Y_val,y_pred)
                c.set_description(str(candidate)+':'+str(score)+'\nBest:'+ str(bestscore))
                c.refresh() 
                sleep(1.0)
                tuners[candidate].record(parameters, score)
                if score > bestscore:
                    bestscore = score
                    bestmodel = candidate
                    best_params = parameters
                if candidate in models_det.keys():
                    if scores[candidate]<score : 
                        self.all_errors[candidate]=er.copy()
                        models_det[candidate]=model
                        scores[candidate]=score
                else:
                    self.all_errors[candidate]=er.copy()
                    models_det[candidate]=model
                    scores[candidate]=score
    
        else : 
            raise ValueError("Error Type of Selection : {} not in ['brute_force','btb']".format(type))
        
        self.best_model=models_det[bestmodel]
        self.models_det=models_det
        self.Bmodel_name=bestmodel
        self.best_param=models_det[bestmodel].hyperparameters
        self.fitted=True
        self.scores=scores
        self.bestscore=bestscore
        print("\n________________________________________")
        print('\n Search for best model has Succesfully finished')
        print('_______')
        print(' Best model : {} '.format(bestmodel))
        print("\n________________________________________")
        return models_det[bestmodel]

    def best_estimator(self):
        if self.fitted:
            return self.best_model
        else : 
            print ("You must call BestModelK() function before ")

    def best_params(self):
        if self.fitted:
            return self.best_param
        else : 
            print ("You must call BestModelK() function before ")

    def print_results(self):
        table = [ [k,v]  if v!=self.bestscore 
                  else [k+' (BEST)', str(v)+' (BEST)'] 
                  for k, v in self.scores.items() ]
        headers = ["Algorithms", self.metric_dict[self.judgment_metric]]
        print(tabulate(table, headers, tablefmt="grid"))
        print('\n Best model : {} '.format(self.Bmodel_name))
        print('_______')
        self.best_model=self.models_det[self.Bmodel_name]
        self.best_param=Algorithm.get_params(self.models_det[self.Bmodel_name])
        print(' Best HyperParameters : {} '.format(self.best_param))
        print('_______')
        print(' Metric: {} '.format(self.metric_dict[self.judgment_metric]))
        print('_______')
        print(' Best Score: {} '.format(self.bestscore))
    
    
    def get_model_V_error(self,models=[],metrics=[]):
        table=[]
        if len(models)==0 and len(metrics)==0:
            for d in self.all_errors.keys():
                table.append([d]+ [round(x,2) for x in list(self.all_errors[d].values())] )
            headers = ["Algorithms"]+ list(self.all_errors[self.Bmodel_name].keys())
        else:
            for d in self.all_errors.keys():
                if d in models:
                    l=[]
                    for mtr in metrics :
                        for m,v in self.all_errors[d].items():
                            if m==mtr:
                                l.append(v)
                    table.append([d]+l )
            if len(metrics)<=3:
                metrics_names=[]
                for e in metrics : 
                    metrics_names.append(self.metric_dict[e])
            else :
                metrics_names=metrics
            headers = ["Algorithms"]+ metrics_names
        print(tabulate(table, headers, tablefmt="grid"))

    def get_model(self,name=None):
        return self.models_det[name]
