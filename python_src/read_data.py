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
        "Date", "Laboratory.ID", "Client.ID","Reading.1",
        "Reading.2", "Escherichia.coli", "Units", "Sample.Collection.Time"
        ]

    for i, sheet_name in enumerate(xls.sheet_names):
        if not xls.book.sheet_by_name(sheet_name).nrows:
            # Older versions of ExcelFile.parse threw an error if the sheet
            # was empty, explicitly check for this condition.
            continue
        df = xls.parse(sheet_name)

        if i == 0 and len(df.columns) > 30:
            # This is the master/summary sheet
            continue

        # Insert name of sheet as first column, the sheet name is the date
        df.insert(0, u'Date', sheet_name)

        for c in df.columns.tolist():
            if 'Reading' in c:
                # There are about 10 days that have >2 readings for some reason
                if int(c[8:]) > 2:
                    df.drop(c, 1, inplace=True)

        if df.index.dtype == 'object':
            # If the first column does not have a label, then the excel
            # parsing engine will helpfully use the first column as
            # the index. This is *usually* helpful, but there are two
            # days when the first column is missing the typical label
            # of 'Laboratory ID'. In this case, peel that index off
            # and set its name.
            df.reset_index(inplace=True)
            df.columns = ['Laboratory ID'] + df.columns.tolist()[1:]

        # Only take the first 8 columns, some sheets erroneously have >8 cols
        df = df.ix[:,0:8]

        # Standardize the column names
        df.columns = standardized_col_names

        dfs.append(df)

    df = pd.concat(dfs)

    df.insert(0, u'Year', year)

    df.dropna(subset=['Client.ID'], inplace=True)

    return df


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


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

    return pd.concat(dfs)

if __name__ == '__main__':
    df = read_data()
    print(df.head())
    print(df.shape)
