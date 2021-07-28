import os

#abstract loader class
class Loader:
    def __init__(self, loader):
        self.loader = loader
    
    def load(self, resource_name):
        return self.loader.load(resource_name)

    def annotate(self):
        pass

    def save(self, dataset, resource_name):
        self.loader.save(dataset, resource_name)