import numpy as np 
import pandas as pd
from .utils import Utils
from sklearn.metrics import (median_absolute_error,mean_absolute_error,r2_score,mean_squared_log_error,
                            mean_squared_error,explained_variance_score,f1_score,precision_score,
                            recall_score,log_loss,average_precision_score,roc_auc_score,accuracy_score,cohen_kappa_score
                            )



class BinaryClassifierScorer:   
    def __init__(self):
        pass

    @classmethod
    def error(cls,y_true, y_predicted):
        BA,BER=Utils.get_BA_BER(y_true, y_predicted)
        Scores={
            "ACCURACY"     : accuracy_score(y_true, y_predicted),
            "ROC_AUC"    : roc_auc_score(y_true, y_predicted),
            "F1"    : f1_score(y_true, y_predicted),
            "AP"   : average_precision_score(y_true, y_predicted) ,
            "PRECISION"  : precision_score(y_true, y_predicted),
            "RECALL"    : recall_score(y_true, y_predicted),
            "NLL"   : -1*log_loss(y_true, y_predicted),
            "BA":BA ,
            "BER" :BER
        }
        return Scores

class MultiClassClassifierScorer:   
    def __init__(self):
        pass
    
    @classmethod
    def error(cls,y_true, y_predicted):
        BA,BER=Utils.get_BA_BER(y_true, y_predicted)
        Scores={
            "ACCURACY"     : accuracy_score(y_true, y_predicted),
            "ROC_AUC_MICRO"    : roc_auc_score(y_true, y_predicted,average='micro'),
            "ROC_AUC_MACRO"   : roc_auc_score(y_true, y_predicted,average='macro'),
            "F1"    : f1_score(y_true, y_predicted),
            "F1_MICRO"     : f1_score(y_true, y_predicted,average='micro'),
            "F1_MACRO"    : f1_score(y_true, y_predicted,average='macro'),
            "F1_WEIGHTED"    : f1_score(y_true, y_predicted,average='weighted'),
            "PRECISION"  : precision_score(y_true, y_predicted),
            "PRECISION_MICRO"  : precision_score(y_true, y_predicted,average='micro'),
            "PRECISION_MACRO"  : precision_score(y_true, y_predicted,average='macro'),
            "PRECISION_WEIGHTED"  : precision_score(y_true, y_predicted,average='weighted'),
            "RECALL"    : recall_score(y_true, y_predicted),
            "RECALL_MICRO"    : recall_score(y_true, y_predicted,average='micro'),
            "RECALL_MACRO"    : recall_score(y_true, y_predicted,average='macro'),
            "RECALL_WEIGHTED"    : recall_score(y_true, y_predicted,average='weighted'),
            "COHEN_KAPPA_SCORE":cohen_kappa_score(y_true, y_predicted),
            "BA":BA ,
            "BER" :BER
        } 
        return Scores




class RegressionScorer:   
    def __init__(self):
        pass
    
    @classmethod
    def error(cls,y_true, y_predicted):
        if  (pd.Series(y_true > 0).all()) and (pd.Series(y_predicted > 0).all())  :
            MSLE_error= mean_squared_log_error(y_true, y_predicted)
            RMSLE_error=mean_squared_log_error(y_true, y_predicted)**0.5
        else: 
            MSLE_error=np.nan 
            RMSLE_error=np.nan 
        Scores={
            "EV"     : explained_variance_score(y_true, y_predicted),
            "MAE"    : -1*mean_absolute_error(y_true, y_predicted),
            "MSE"    : -1*mean_squared_error(y_true, y_predicted),
            "MSLE"   : -1*MSLE_error ,
            "MeAE"   : -1*median_absolute_error(y_true, y_predicted),
            "R2"     : r2_score(y_true, y_predicted),
            "RMSE"   : -1*mean_squared_error(y_true, y_predicted)**0.5,
            "RMSLE"  : -1*RMSLE_error
        }
        return Scores

    @classmethod
    def report(cls,y_true,y_predicted, chuncks=5):
        actuals = list(y_true)
        predictions = list(y_predicted)
        Utils.differences(predictions=predictions, actuals=actuals)

        actuals_preds = list(zip(actuals, predictions))
        actuals_preds.sort(key=lambda pair: pair[1])
        actuals_sorted = [act for act, pred in actuals_preds]
        predictions_sorted = [pred for act, pred in actuals_preds]

        n=100//chuncks
        print('\nThe trained estimator performance on each successive {}% of the predictions:'.format(n))
        for i in range(1,chuncks+1):
            print('\n___________')
            print('chunk:')
            print(i)
            min_idx = int((i - 1) / chuncks * len(actuals_sorted))
            max_idx = int(i / chuncks * len(actuals_sorted))
            actuals_for_this_chunck = actuals_sorted[min_idx:max_idx]
            predictions_for_this_chunck = predictions_sorted[min_idx:max_idx]
            print('\nAverage predicted values in this chunck : ')
            print(sum(predictions_for_this_chunck) * 1.0 / len(predictions_for_this_chunck))
            print('\nAverage actual values in this chunck : ')
            print(sum(actuals_for_this_chunck) * 1.0 / len(actuals_for_this_chunck))
            print('\nRMSE for this chunck : \n')
            print(mean_squared_error(actuals_for_this_chunck, predictions_for_this_chunck)**0.5)
            Utils.differences(predictions_for_this_chunck, actuals_for_this_chunck)
       
        