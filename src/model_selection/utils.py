import numpy as np 

class Utils:   
    def __init__(self):
        pass
    
    @classmethod
    def get_BA_BER(cls,y_true, y_predicted):
        BER = []
        BA=[]
        epsilon = 1e-15
        for i in np.unique(y_true):
            tp = ((y_true == i) & (y_predicted == i)).sum()
            tn = ((y_true != i) & (y_predicted != i)).sum()
            fp = ((y_true != i) & (y_predicted == i)).sum()
            fn = ((y_true == i) & (y_predicted != i)).sum()
            BER.append(1 - 0.5*(tp / np.maximum(tp + fn, epsilon)) - 0.5*(tn / np.maximum(tn + fp, epsilon)))
            BA.append((fp+fn)/(tp+tn+fp+fn))
        return np.mean(BA),np.mean(BER)

    @classmethod
    def differences(cls,predictions, actuals):
        pos_differences = []
        neg_differences = []
        for idx, pred in enumerate(predictions):
            difference = pred - actuals[idx]
            if difference > 0:
                pos_differences.append(difference)
            elif difference < 0:
                neg_differences.append(difference)
        print('\nCount of positive differences (prediction > actual):')
        print(len(pos_differences))
        print('\nCount of negative differences:')
        print(len(neg_differences))
        if len(pos_differences) > 0:
            print('\nAverage positive difference:')
            print(sum(pos_differences) * 1.0 / len(pos_differences))
        if len(neg_differences) > 0:
            print('\nAverage negative difference:')
            print(sum(neg_differences) * 1.0 / len(neg_differences))

    @classmethod
    def list_to_dict(cls,list=None):
        d=dict()
        for e in list:
            if type(e)!=str:
                d[e.__class__.__name__]=-9999
            else : 
                d[e]=-9999
        return d
    
    @classmethod
    def _import(cls,name):
        name=name.replace('<', '')
        name=name.replace('>', '')
        name=name.replace('class', '')
        name=name.replace('\'', '')
        name=name.strip()
        components = name.split('.')
        mod = __import__(components[0])
        for comp in components[1:]:
            mod = getattr(mod, comp)
        return mod
     
    @classmethod
    def get_columns_indices(cls,df_columns=None,col_names=None):
        indices=[]
        if col_names is None :
            return None
        for e in col_names:
            indices.append(df_columns.index(e))
        return indices

    @classmethod
    def get_count(cls,dict_models=None,name=None):
        count=1
        for e in dict_models:
            if e[:len(name)]==name : 
                count=count+1
        return count