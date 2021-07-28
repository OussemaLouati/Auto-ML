import pandas as pd
import math
import numpy as np
import statistics
from collections import OrderedDict

class Binning():
    """
        Apply Binning on noisy numerical attributes.
    """
    def __map_binning_to_data(self,binn,X_dict,nb_of_data_in_bin):
        i = 0
        j = 0
        x_new = []
        for _ in range(len(X_dict)):
            if(i<nb_of_data_in_bin):
                print(binn[j])
                x_new.append(np.round(binn[j], 3))
                i = i + 1
            else:
                i = 0
                j = j + 1
                x_new.append(np.round(binn[j], 3))
                i = i + 1
        return x_new

    def mean_binning(self,bin_size,x):
        """
            Apply mean binning.
            Parameters:
             --------
                • bin_size (int)
                • x (array): - Refers to the column.
            Returns:
                • list : -The size is the same the parameter x.
        """
        binn =[]
        # a variable to store the mean of each bin
        avrg = 0
        #number of variables in each bin
        nb_of_data_in_bin = int(math.ceil(len(x)/bin_size))
        # X_dict will store the data in sorted order
        X_dict = OrderedDict()
        for i in range(len(x)):
            X_dict[i]= x[i]
        i = 0
        k = 0
        for _, val in X_dict.items():
            if(i<nb_of_data_in_bin):
                avrg = avrg + val
                i = i + 1
            elif(i == nb_of_data_in_bin):
                k = k + 1
                i = 0
                binn.append(np.round((avrg / nb_of_data_in_bin), 3))
                avrg = 0
                avrg = avrg + val
                i = i + 1
        rem = len(x) % bin_size
        if rem == 0 :
            binn.append(np.round((avrg / nb_of_data_in_bin), 3))
        else:
            binn.append(np.round(avrg / rem, 3))
        #Apply binning to data
        return self.__map_binning_to_data(binn,X_dict,nb_of_data_in_bin)

    def median_binning(self,bin_size,x):
        """
            Apply median binning.
            Parameters:
             --------
                • bin_size (int)
                • x (array): - Refers to the column.
            Returns:
                • list : -The size is the same the parameter x.
        """
        binn =[]
        # a variable to store the mean of each bin
        avrg =[]
        #number of variables in each bin
        nb_of_data_in_bin = int(math.ceil(len(x)/bin_size))
        # X_dict will store the data in sorted order
        X_dict = OrderedDict()
        for i in range(len(x)):
            X_dict[i]= x[i]
        i = 0
        k = 0
        # performing binning
        for _, val in X_dict.items():
            if(i<nb_of_data_in_bin):
                avrg.append(val)
                i = i + 1
            elif(i == nb_of_data_in_bin):
                k = k + 1
                i = 0
                binn.append(statistics.median(avrg))
                avrg =[]
                avrg.append(val)
                i = i + 1
        binn.append(statistics.median(avrg))
        #Apply binning to data
        return self.__map_binning_to_data(binn,X_dict,nb_of_data_in_bin)

    def boundary_binning(self,bin_size,x):
        """
            Apply boundary binning.
            Parameters:
             --------
                • bin_size (int)
                • x (array): - Refers to the column.
            Returns:
                • list : -The size is the same the parameter x.
        """
        binn =[]
        # variable to store the mean of each bin
        avrg =[]
        #number of variables in each bin
        nb_of_data_in_bin = int(math.ceil(len(x)/bin_size))
        # X_dict will store the data in sorted order
        X_dict = OrderedDict()
        for i in range(len(x)):
            X_dict[i]= x[i]
        i = 0
        k = 0
        # performing binning
        for _, val in X_dict.items():
            if(i<nb_of_data_in_bin):
                avrg.append(val)
                i = i + 1
            elif(i == nb_of_data_in_bin):
                k = k + 1
                i = 0
                binn.append([min(avrg), max(avrg)])
                avrg =[]
                avrg.append(val)
                i = i + 1
        binn.append([min(avrg), max(avrg)])
        #Apply binning to data
        return self.__map_boundary_bin_to_data(binn,X_dict,nb_of_data_in_bin)

    def __map_boundary_bin_to_data(self,binn,X_dict,nb_of_data_in_bin):
        # variable initiation
        x_new = []
        i=0
        j=0
        for _, val in X_dict.items():
            if(i<nb_of_data_in_bin):
                if(abs(val-binn[j][0]) >= abs(val-binn[j][1])):
                    x_new.append(binn[j][1])
                    i = i + 1
                else:
                    x_new.append(binn[j][0])
                    i = i + 1
            else:
                i = 0
                j = j + 1
                if(abs(val-binn[j][0]) >= abs(val-binn[j][1])):
                    x_new.append(binn[j][1])
                else:
                    x_new.append(binn[j][0])
                i = i + 1
        return x_new
