from speedml import Speedml
import numpy as np
import os
from datetime import datetime
import json
import pandas as pd
import re
from prettytable import PrettyTable
from shutil import copyfile
import webbrowser

class Overviewer():
    """
    This is a class for data overviewer and data annotation.\\
    Note that the predefined annotation are : \\
        "KPI","Geo", "Latitude", "Longitude", "Altitude", "Datetime" and "Date"
    """
    annotation = {
                    1:"KPI",
                    2:"Geo",
                    3:"Latitude",
                    4:"Longitude",
                    5:"Altitude",
                    6:"Datetime",
                    7:"Date" }
    description_cols = ['Numerical High-cardinality', 'Numerical Categorical',
       'Numerical Continuous', 'Text High-cardinality', 'Text Categorical']
    def __init__(self,data,target,index_col=None):
        """
        Constructor for class Overviewer
            Parameters:
                • data : pandas dataframe
                • target: string, refers to the target column name
                • index_col: string,
        """
        #Test if there's an index in the dataset
        varr = True
        #Put the index name into lowercase format to avoid confusion
        for cols in data.columns:
            if "index" == cols.lower:
                data.rename(columns={cols: "index"},inplace = True)
                varr = False
        #Create an index if var == True
        if varr:
            data.reset_index(inplace=True)
        self.data=data
        self.backup_data=data
        self.dic=dict()
        cwd = os.getcwd()
        path = os.path.dirname(os.path.abspath(__file__))
        copyfile(path+"/myStyle.css", cwd+"myStyle.css")
        self.target=target
        self.path=cwd
        self.path_train= cwd+"train.csv"
        self.path_test= cwd+"test.csv"
        self.eda=pd.DataFrame()

    def output(self):
        """
            Return:
                • dict: the key is the column name and the value is composed of the type, description and annotation.
        """
        #Call the dtype_column method to have the new typed dataframe
        self.data = self.dtype_columns()
        #Prepare the format for the speedml params
        msk = np.random.rand(len(self.data)) < 0.8
        self.data[msk].to_csv(self.path+"train.csv")
        self.data[~msk].to_csv(self.path+"test.csv")
        #Speedml initiation
        sml = Speedml(self.path_train,
              self.path_test,
              target=self.target, uid='index')
        sml.configure("unique_ratio",10)
        self.eda=sml.eda()
        #Drop the folder where  created train and test for the speedml
        if os.path.exists(self.path):
            os.remove(self.path+"train.csv")
            os.remove(self.path+"test.csv")
        return self.eda

    def report(self):
        """
        This function creates a HTML report containing an overview of the dataset from //
        statistical information, data types, value counts, the head and tail of the data.
        """
        df = Overviewer.data_characterization(self)
        data_old = self.data
        data_old = data_old.drop(["index"],axis=1)
        #create folder
        cwd = os.getcwd()
        if not os.path.exists(cwd):
            os.mkdir(cwd)
        #Download data to csv to open them the right form for PrettyTable
        df.to_csv(cwd+"data_characterization.csv",index = False) #data caracterization df
        data_old.to_csv(cwd+"data.csv",index = False) #input data df
        #Open Files
        file_data = open(cwd+"data_characterization.csv",'r')
        data_old_ = open(cwd+"data.csv",'r')
        #Read data
        data_old_v1 = data_old_.readlines()
        data = file_data.readlines()
        #Build the data
        data_rows = []
        for i in range(0,len(df)+1):
            data_rows.append(data[i].split(","))
        #Table for description
        x = PrettyTable(list(df.columns))
        x.format = True
        for a in range(1,len(data_rows)):
            row = [data_rows[a][i] for i in range(0,df.shape[1])]
            x.add_row(row)
        #Table for head
        y = PrettyTable(list(data_old.columns))
        data_rows = []
        for i in range(0,6):
            data_rows.append(data_old_v1[i].split(","))
        for a in range(1,len(data_rows)):
            row = [data_rows[a][i]  for i in range(0,data_old.shape[1])]
            y.add_row(row)
        #Table for tail
        data_rows = []
        for i in range(len(data_rows),len(data_rows)-6,-1):
            data_rows.append(data_old_v1[i].split(","))
        z = PrettyTable(list(data_old.columns))
        for a in range(1,len(data_rows)):
            row = [data_rows[a][i]  for i in range(0,data_old.shape[1])]
            z.add_row(row)
        #Create html report
        head = """  <head>  <meta charset = "UTF-8">
                    <title> Data Overview Report </title>
                    <link rel = "stylesheet"
                    type = "text/css"
                    href = """+'"'+cwd+"myStyle.css"+"""" />
                     </head>
                """
        htmlfile = open(cwd+"report.html",'w')
        htmlfile.write(head)
        htmlfile.write("""<h1>Data Overview Report</h1>""")
        htmlfile.write("""<h3>Rows: """+str(self.data.shape[0])+""", Columns: """+str(df.shape[0])+"""</h3>""")
        html = x.get_html_string(attributes={"name":"my_table", "class":"report"})
        htmlfile.write(html)
        htmlfile.write("""<h1>Data Head</h1>""")
        html_head = y.get_html_string(attributes={"name":"my_table", "class":"report"})
        htmlfile.write(html_head)
        htmlfile.write("""<h1>Data Tail</h1>""")
        html_tail = z.get_html_string(attributes={"name":"my_table", "class":"report"})
        htmlfile.write(html_tail)
        #Drop the folder and the files after usage
        file_data.close()
        data_old_.close()
        #Remove files
        if os.path.exists(cwd):
            os.remove(cwd+"data_characterization.csv")
            os.remove(cwd+"data.csv")
        url = 'file:///'+cwd+"report.html"
        webbrowser.open(url, new=2)

    def data_characterization(self):
        """
            Give statistical insight of the data in a dataframe where the index is the column name.
            Return:
                • pandas dataframe: -containning statistical information for each column of the dataset.
        """
        Overviewer.output(self)
        data = self.data
        data = data.drop(["index"],axis=1)
        df = pd.DataFrame()
        Count = []
        final_value_count = []
        Nan_counts = data.isnull().sum().tolist()
        missing_vals = data.isnull().sum()
        Nan_ratio = []
        missing_val_percent = 100 * missing_vals / data.shape[0]
        columns = data.columns
        self.insert_annotation_json({})
        data_description = data.describe()
        statistical_info = data_description.loc[data_description.index[1:]].T
        statistical_info.reset_index(inplace = True)
        statistical_info.rename(columns={'index':'Columns_name'},inplace=True)
        #Attribute for each column the % of missing values
        for  col  in columns :
            for index,val in zip(missing_val_percent.index,missing_val_percent):
                 if index==col :
                     Nan_ratio.append(val)
                     continue
            #Value count
            i = 0
            value_count = data[col].value_counts()
            value_counts = []
            #Store top 5 values for each column based on their occurence
            for val, occurence in zip(value_count.index,value_count.values):
                if i<=5:
                    value_counts.append(str(val)+":"+str(occurence))
                    i += 1
                else:
                    break
            value_counts_String=""
            for val in value_counts:
               value_counts_String = value_counts_String + val + " "
            final_value_count.append(value_counts_String)
            Count.append(len(list(data[col].unique())))
        df["Columns_name"] = columns
        df["Description"] = [self.dic.get(i)[1].get("description") for i in columns]
        df["Type"] = data.dtypes.tolist()
        df["Nb_unique_values"] = Count
        df["Nb_Nan_values"] = Nan_counts
        df["%_Nan_values"] = Nan_ratio
        df["Unique_values(value:count)"] = final_value_count
        df = df.merge(statistical_info,on="Columns_name",how="left")
        df.fillna("-",inplace=True)
        return df

    def insert_annotation_json(self,annotations={}):
        """
            Attributes annotations to the data columns in a dictionnary to be used in the data exploration.
            Parameters:
             --------
                • annotations(dict): - dictionnary where the key is the \\
                column name and the value is the annotation(s)
            Return:
                • dict: the key is the column name and the value is composed of the type, description and annotation.
        """
        existing_annotations = ""
        for _, annotation in annotations.items():
            if annotation not in self.annotation.values():
                existing_annotations="-".join(self.annotation.values())
                raise Exception("Annotation not defined in the existing annotations: ["+existing_annotations+"]")
        #make sure that the type is datetime for the date annotations
        for col_name in self.data.columns:
            if (annotations.get(col_name)=="Datetime") | (annotations.get(col_name)=="Date"):
                self.data[col_name] =  self.data[col_name].astype(('datetime64[ns]'))
        #Create the dictionnary for the JSON file
        for col_name,col_type in zip(self.data.columns,self.data.dtypes):
            l=list()
            l.append({"type":str(col_type)})
            for description in Overviewer.description_cols:
                try:
                    items=self.eda.iloc[self.eda.index==description, 0]
                    if col_name in items[0]:
                        l.append({"description":description})
                except:
                    pass
            l.append({"annotation":annotations.get(col_name)})
            self.dic[col_name]=l
        return self.dic

    #Convert an object like currency or percentage to float
    def __currency_percent_to_float(self,df,colname):
        return [float(re.sub(r'[^\d.-]', '', val)) for val in df[colname]]

    def dtype_columns(self):
        """
            Return:
                • pandas dataframe: - dataframe with optimized column types.
        """
        df=self.data
        for col in df.columns:
            col_type = df[col].dtype
            #Test if the object is a currency or percentage then covert it to float
            try:
                if (col_type == object):
                    if([True for i in df[col][0] if i in ["$","%","£",","]]):
                        df[col]=self.__currency_percent_to_float(df,col)
                        col_type = df[col].dtype
            except:
                continue
            #Convert fat types of integer and float
            if (col_type != object)&(col_type.name!="category")&(col_type != '<M8[ns]'):
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
            # Attribute category as the new type for object column
            elif  (col_type == object) :
                df[col] = df[col].astype('category')
        return df
