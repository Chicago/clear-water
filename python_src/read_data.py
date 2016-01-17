from __future__ import print_function
import pandas as pd
import os

"""
This file reads in data related E. coli levels
in Chicago beaches.
"""

def split_sheets(file_name, year):
    '''
    Reads in all sheets of an excel workbook, concatenating
    all of the information into a single dataframe.

    The excel files were unfortunately structured such that
    each day had its own sheet.
    '''
    xls = pd.ExcelFile(file_name)
    dfs = []
    standardized_col_names = [
        ".id", "Laboratory.ID", "Client.ID","Reading.1",
        "Reading.2", "Escherichia.coli", "Units", "Sample.Collection.Time"
        ]
    for sheet_name in xls.sheet_names:
        if not xls.book.sheet_by_name(sheet_name).nrows:
            # Older versions of ExcelFile.parse threw an error if the sheet
            # was empty, explicitly check for this condition.
            continue
        df = xls.parse(sheet_name)

        # Insert name of sheet as first column
        df.insert(0, u'.id', sheet_name)

        # Only take the first 8 columns, some sheets erroneously have >8 cols
        df = df.ix[:,0:8]

        # Standardize the column names
        df.columns = standardized_col_names

        # Remove all rows where Client ID is NaN
        df.dropna(subset=['Client.ID'], inplace=True)

        dfs.append(df)

    if dfs[0].shape[0] > 30:
        # gets rid of summary or master sheet
        dfs = dfs[1:]

    return pd.concat(dfs)


def read_data():
    '''
    Read in the excel files for years 2006-2015 found in
    'data/ChicagoParkDistrict/raw/Standard 18 hr Testing'
    '''

    data_path = "../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/"
    data_path = os.path.join(os.path.dirname(__file__), data_path)

    df2006 = split_sheets(data_path + "2006 Lab Results.xls", 2006)
    df2007 = split_sheets(data_path + "2007 Lab Results.xls", 2007)
    df2008 = split_sheets(data_path + "2008 Lab Results.xls", 2008)
    df2009 = split_sheets(data_path + "2009 Lab Results.xls", 2009)
    df2010 = split_sheets(data_path + "2010 Lab Results.xls", 2010)
    df2011 = split_sheets(data_path + "2011 Lab Results.xls", 2011)
    df2012 = split_sheets(data_path + "2012 Lab Results.xls", 2012)
    df2013 = split_sheets(data_path + "2013 Lab Results.xls", 2013)
    df2014 = split_sheets(data_path + "2014 Lab Results.xls", 2014)
    df2015 = split_sheets(data_path + "2015 Lab Results.xlsx", 2015)

    dfs = [
        df2006, df2007, df2008, df2009,df2010,
        df2011, df2012, df2013, df2014, df2015
        ]

    # Combine Data
    return pd.concat(dfs)

if __name__ == '__main__':
    df = read_data()
