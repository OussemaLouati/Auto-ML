import pandas as pd
from .loader import Loader

'''
"resource_name" must reference a full path to source/target files in this loader
'''

class LocalFileLoader(Loader):
    def __init__(self):
        pass

    def load(self, resource_name):
        return pd.read_csv(resource_name)

    def save(self, dataset, resource_name):
        dataset.to_csv(resource_name, index=False)