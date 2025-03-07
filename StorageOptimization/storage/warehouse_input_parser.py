import pandas as pd
from os.path import isfile

class WarehouseInputParser(object):

    @staticmethod
    def try_to_make_a_df(indata):
        if not isfile(indata):
            try:
                return pd.from_dict(indata)
            except:
                pass
        try:
            return pd.read_csv(indata)
        except:
            pass
        try:
            return pd.read_excel(indata, sheet_name=0)
        except:
            raise ValueError('Unable to generate a dataframe from input!')


    @staticmethod
    def sanitize_string(string):
        return ''.join(filter(str.isalnum, string)).lower()


    @staticmethod
    def sanitize_warehouse_input(self,infile=None):

        if infile is None:
            infile = self.infile

        if infile.endswith('csv'):
            # try:
            indf = pd.read_csv(infile)
            # except:
            #     raise InvalidWarehouseInputError()
        else:
            indf = pd.read_excel(infile, sheet_name=1)



        # sanitized the column names
        input_colnames = [ self.sanitize_string(col) for col in indf.columns ]
        required_columns = self.sanitized_required_columns

        missing_columns = set(required_columns)-set(input_colnames)
        excess_columns = set(input_colnames)-set(required_columns)

        if missing_columns:
            raise ValueError(f'Expected the following column names in warehoue input file:\n{self.required_columns}; got {self.input_colnames}')

        return pd.DataFrame(columns = input_colnames)


    ####################
    #
    #   constructor
    #
    ####################
    def __init__(self,
                 infile,
                 *args,
                 **kwargs):

        self.infile = infile
        self.input_data = WarehouseInputParser.try_to_make_a_df(infile)
        self.required_columns = [   'ShelfBay',
                                    'ShelfType',
                                    'ShelfColumn',
                                    'ShelfAisle',
                                    'ShelfPriority' ]
        self.lowercase_required_colnames = [ colname.lower() for colname in self.required_columns ]
        self.sanitized_required_columns = [ self.sanitize_string(colname).lower() \
                                           for colname in self.lowercase_required_colnames]



    def process_warehouse_input(self,
                    verbose = False,
                    indata: pd.DataFrame = None): # indata should be sanitized

        if indata is None:
            indata = self.input_data
        elif isinstance(indata, str):
            indata = self.try_to_make_a_df(indata)


        n_rows_input = len(indata)
        # expect 3xn_rows_input as size of output


        outdfs = []

        for i, compart in enumerate('ABCD'): # shelf levels
            outdf = pd.DataFrame(columns = list(indata.columns)+[ 'Compartment', 'ShelfID' ])
            for idx, row in indata.iterrows():
                for colname in indata.columns:
                    outdf.at[idx, colname] = indata.at[idx, colname]
                    outdf.at[idx, 'Compartment'] = compart
                    outdf.at[idx, 'ShelfID'] = (row.ShelfColumn, row.ShelfAisle, compart)

                    if row.ShelfType=='Adjustable':
                        outdf.at[idx, 'ShelfLength'] = 56
                        outdf.at[idx, 'ShelfWidth'] = 59
                        outdf.at[idx, 'ShelfHeight'] = int(row.AdjustableConfig.strip().split('-')[i])*12
                        outdf.at[idx, 'WeightCapacity'] = 2000/3
                    elif row.ShelfType=='Standard':
                        outdf.at[idx, 'ShelfLength'] = 9*12//2
                        outdf.at[idx, 'ShelfWidth'] = 42
                        outdf.at[idx, 'ShelfHeight'] = 60
                        outdf.at[idx, 'WeightCapacity'] = 5060/2

            outdfs.append(outdf)

        self.outdata = pd.concat(outdfs)\
                         .sort_values(by=['ShelfColumn','ShelfAisle'])\
                         .reset_index()\
                         .drop(columns=['index'])

        return self.outdata
