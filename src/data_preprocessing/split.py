from sklearn.model_selection import StratifiedShuffleSplit,StratifiedKFold
import numpy as np

class Split():
    """
        This class is for stratified splitting.
        Attributes:
        --------
            • X (ndarray): - Represents the features of the dataset.
            • y (array): - Represent the target column.
    """
    def __init__(self,X,y):
        """
         Parameters:
             --------
                • X (ndarray): - Represents the features of the dataset.
                • y (array): - Represent the target column.
        """
        self.X = X
        self.y = y

    def __split(self,split):
        for train_index, test_index in split.split(self.X,self.y):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
        return X_train, X_test, y_train, y_test

    def stratifiedShuffleSplit(self,n_split=5,test_size=0.3,random_state=0):
        """
            Apply cleaning techniques on text categorical values.
            Parameters:
             --------
                • n_split (int): - Number of splits
                • test_size (float): - The size of the test sample for the dataset.
                • random_state (int): - Random number.
            Returns:
                • X_train (ndarray), X_test(array), y_train (ndarray), y_test(array)
        """
        split = StratifiedShuffleSplit(n_splits=n_split, test_size=test_size, random_state=random_state)
        return Split.__split(self,split)

    def stratifiedKfold(self,n_split=5,random_state=0):
        """
            Apply stratified Kfold splitting of the data.
            Parameters:
             --------
                • n_split (int): - Number of splits
                • random_state (int): - Random number.
            Returns:
                • X_train (ndarray), X_test(array), y_train (ndarray), y_test(array)
        """
        split = StratifiedKFold(n_splits=n_split, random_state=random_state,shuffle=True)
        return Split.__split(self,split)
