**Data overview module** (`overviewer`): Provides a `Overviewer` class  that include:

`top(N_row)`: preview top N rows (default 5 rows)

`tail(N_row)`: preview buttom N rows (default 5 rows)

`ouput()`: gives information about column type, row_count, model and suggest plots. 
It calls also `dtype_columns()` function to attribute data-types.  

`details()`: preview occurrence values per column, statistics details(mean, max, mode.. ), columns types.

`configure()`: changes a parameter of Speedml if needed.

`insert_annotation_json(annotations)`: create a JSON file descriping the columns of the data, defining their types and attributing annotations. 

    ##### Example:

    ```
        #Takes as parameter a Pandas dataframe and the target name column
        overview = Overviewer(dataframe, "target")
       
       #Takes as a parameter number of rows to preview, if not mentioned (none) previews 5 rows.
        overview.top(N=None)

        #Takes as a parameter number of rows to preview, if not mentioned (none) previews 5 rows.
        overview.tail(N=None)

        #Requires no parameter since we instantiated the overviewer with the data
        overview.output()

        #Take as parameter the name of the paramter and the value
        overview.configure("high_cardinality",15)

        #Recall to observe the change
        overview.output()

        #Requires no parameter
        overview.details()
        
        #Takes as parameter a dictionnary where the key is the column name and the parameter is the annotation
        overview.insert_annotation_json({col_name1:"annotation1",col_name2:"annotation2"})
    ```

<br>

<br>

