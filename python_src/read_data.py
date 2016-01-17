from __future__ import print_function
import pandas as pd
import os
import matplotlib.pyplot as plt

'''
This file reads in data related E. coli levels
in Chicago beaches. It is based on the files
analysis.R and split_sheets.R, and is written
such that the dataframe loaded here will match
the R dataframe code exactly.
'''

# TODO: verbose
# TODO: plot toggle
# TODO: use multi-level index on date/beach
# TODO: standardize on inplace=True or not inplace

TO_PLOT = True

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
        'Date', 'Laboratory.ID', 'Client.ID','Reading.1',
        'Reading.2', 'Escherichia.coli', 'Units', 'Sample.Collection.Time'
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

        if df.index.dtype == 'object':
            # If the first column does not have a label, then the excel
            # parsing engine will helpfully use the first column as
            # the index. This is *usually* helpful, but there are two
            # days when the first column is missing the typical label
            # of 'Laboratory ID'. In this case, peel that index off
            # and set its name.
            df.reset_index(inplace=True)
            df.columns = ['Laboratory ID'] + df.columns.tolist()[1:]

        # Insert name of sheet as first column, the sheet name is the date
        df.insert(0, u'Date', sheet_name)

        for c in df.columns.tolist():
            if 'Reading' in c:
                # There are about 10 days that have >2 readings for some reason
                if int(c[8:]) > 2:
                    df.drop(c, 1, inplace=True)

        # Only take the first 8 columns, some sheets erroneously have >8 cols
        df = df.ix[:,0:8]

        # Standardize the column names
        df.columns = standardized_col_names

        dfs.append(df)

    df = pd.concat(dfs)

    df.insert(0, u'Year', str(year))

    df.dropna(subset=['Client.ID'], inplace=True)

    return df


def print_full(x):
    '''
    Helper function to plot the *full* dataframe.
    '''
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def date_lookup(s):
    '''
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.

    Thanks to fixxxer, found at
    http://stackoverflow.com/questions/29882573
    '''
    dates = {date:pd.to_datetime(date, errors='ignore') for date in s.unique()}
    for date, parsed in dates.iteritems():
        if type(parsed) is not pd.tslib.Timestamp:
            fmt = '%B %d (%p) %Y'
            dates[date] = pd.to_datetime(date,format=fmt)
    return s.apply(lambda v: dates[v])


def read_data():
    '''
    Read in the excel files for years 2006-2015 found in
    'data/ChicagoParkDistrict/raw/Standard 18 hr Testing'
    along with drekbeach data.

    Also reformats columns in accordance with the transformations
    found in analysis.R
    '''

    data_path = '../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    data_path = os.path.join(os.path.dirname(__file__), data_path)

    dfs = []

    for yr in range(2006,2015):
        dfs.append(split_sheets(data_path + str(yr) +' Lab Results.xls', yr))
    dfs.append(split_sheets(data_path + '2015 Lab Results.xlsx', 2015))

    df = pd.concat(dfs)

    # Need to reset the index to deal with the repeated concatenations
    df.index = range(0, len(df.index))

    # Some records are of the form <1 or >2440
    # Remove the operator and treat the remaining string as the value.
    # Also convert string to float, if possible
    for col in ['Reading.1', 'Reading.2', 'Escherichia.coli']:
        for i, val in enumerate(df[col].tolist()):
            if isinstance(val, basestring):
                val = val.replace('<', '').replace('>', '')
                try:
                    df.ix[i, col] = float(val)
                except ValueError:
                    # Sometimes strings are things like 'Sample Not Received'
                    df.ix[i, col] = float('nan')
        df[col] = df[col].astype('float64')

    # Massage dates, create weekday column
    df.insert(0, 'Full_date',
              df[['Date', 'Year']].apply(lambda x: ' '.join(x), axis=1))
    df['Full_date'] = date_lookup(df['Full_date'])
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df['Weekday'] = df['Full_date'].map(lambda x: days[x.dayofweek])

    # Some header rows were duplicated
    df = df[df['Laboratory.ID'] != u'Laboratory ID']

    # Normalize the beach names
    df['Client.ID'] = df['Client.ID'].map(lambda x: x.strip())
    cleanbeachnames = pd.read_csv(data_path + 'cleanbeachnames.csv')
    cleanbeachnames = dict(zip(cleanbeachnames['Old'], cleanbeachnames['New']))
    # There is one observation that does not have a beach name in the
    # Client.ID column, remove it.
    df = df[df['Client.ID'].map(lambda x: x in cleanbeachnames)]
    df['Client.ID'] = df['Client.ID'].map(lambda x: cleanbeachnames[x])

    # Read in drek beach data
    drek_data_path = '../data/DrekBeach/'
    drek_data_path = os.path.join(os.path.dirname(__file__), drek_data_path)
    drekdata = pd.read_csv(drek_data_path + 'daily_summaries_drekb.csv')
    drekdata.columns = ['Beach', 'Date', 'Drek_Reading',
                         'Drek_Prediction', 'Drek_Worst_Swim_Status']
    drekdata['Date'] = date_lookup(drekdata['Date'])
    drekdata['Beach'] = drekdata['Beach'].map(lambda x: x.strip())
    drekdata['Beach'] = drekdata['Beach'].map(lambda x: cleanbeachnames[x])

    # Merge the data
    df = pd.merge(df, drekdata,
                  left_on=['Client.ID', 'Full_date'],
                  right_on=['Beach', 'Date'],
                  how='left')
    # Both dataframes had a Date column, they got replaced
    # by Date_x and Date_y, drop the drek date and re-name Date_x.
    df.drop('Date_y', 1, inplace=True)
    c = df.columns.tolist()
    c[c.index('Date_x')] = 'Date'
    df.columns = c

    # There was an anamolous reading, the max possible value from the test
    # is around 2420, but one reading was 6488.
    # We need to do the ~(reading 1 > 2500 | reading 2 > 2500) instead of
    # (reading 1 < 2500 & reading 2 < 2500) since the latter returns
    # False if there is a NaN.
    df = df[~((df['Reading.1'] > 2500) | (df['Reading.2'] > 2500))]

    return df

if __name__ == '__main__':
    df = read_data()

    if TO_PLOT:
        normal = df['Reading.1'] + df['Reading.2'] < 235.0 * 2
        arith_not_geo = (
            df['Reading.1'] + df['Reading.2'] >= 235.0 * 2) & (
            (df['Reading.1'] * df['Reading.2']) ** 0.5 < 235
        )
        print(sum(arith_not_geo))
        geo = (df['Reading.1'] * df['Reading.2']) ** 0.5 >= 235

        plt.plot(df.ix[normal, 'Reading.1'],
                 df.ix[normal, 'Reading.2'], '.')
        plt.hold(True)
        plt.plot(df.ix[arith_not_geo, 'Reading.1'],
                 df.ix[arith_not_geo, 'Reading.2'], '.',
                 color=(1, .64, 0))
        plt.plot(df.ix[geo, 'Reading.1'],
                 df.ix[geo, 'Reading.2'], '.',
                 color=(.9, .05, .05))
        plt.title('Reading 1 vs. Reading\nComparing arithmetic and geometric means')

        plt.gca().set_aspect('equal', 'datalim')

        plt.show()
