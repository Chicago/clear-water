from __future__ import print_function
import pandas as pd
import os
import logging
import argparse

'''
This file reads in data related E. coli levels
in Chicago beaches. It is based on the files
analysis.R and split_sheets.R, and is written
such that the dataframe loaded here will match
the R dataframe code exactly.
'''

# TODO: verbose
# TODO: use multi-level index on date/beach ?
# TODO: standardize on inplace=True or not inplace
# TODO: how much consistency do we want between python columns
#       and the R columns?


def split_sheets(file_name, year, verbose=False):
    '''
    Reads in all sheets of an excel workbook, concatenating
    all of the information into a single dataframe.

    The excel files were unfortunately structured such that
    each day had its own sheet.
    '''
    xls = pd.ExcelFile(file_name)
    dfs = []
    standardized_col_names = [
        'Date', 'Laboratory.ID', 'Client.ID', 'Reading.1',
        'Reading.2', 'Escherichia.coli', 'Units', 'Sample.Collection.Time'
    ]

    for i, sheet_name in enumerate(xls.sheet_names):
        if not xls.book.sheet_by_name(sheet_name).nrows:
            # Older versions of ExcelFile.parse threw an error if the sheet
            # was empty, explicitly check for this condition.
            logging.debug('sheet "{0}" from {1} is empty'.format(sheet_name,
                                                                 year))
            continue
        df = xls.parse(sheet_name)

        if i == 0 and len(df.columns) > 30:
            # This is the master/summary sheet
            logging.debug('ignoring sheet "{0}" from {1}'.format(sheet_name,
                                                                 year))
            continue

        if df.index.dtype == 'object':
            # If the first column does not have a label, then the excel
            # parsing engine will helpfully use the first column as
            # the index. This is *usually* helpful, but there are two
            # days when the first column is missing the typical label
            # of 'Laboratory ID'. In this case, peel that index off
            # and set its name.
            msg = '1st column in sheet "{0}" from {1} is missing title'.format(
                sheet_name, year)
            logging.debug(msg)
            df.reset_index(inplace=True)
            df.columns = ['Laboratory ID'] + df.columns.tolist()[1:]

        # Insert name of sheet as first column, the sheet name is the date
        df.insert(0, u'Date', sheet_name)

        for c in df.columns.tolist():
            if 'Reading' in c:
                # There are about 10 days that have >2 readings for some reason
                if int(c[8:]) > 2:
                    logging.info('sheet "{0}" from {1} has >2 readings'.format(
                        sheet_name, year)
                    )
                    df.drop(c, 1, inplace=True)

        # Only take the first 8 columns, some sheets erroneously have >8 cols
        df = df.ix[:,0:8]

        # Standardize the column names
        df.columns = standardized_col_names

        dfs.append(df)

    df = pd.concat(dfs)

    df.insert(0, u'Year', str(year))

    logging.info('Removing data with missing Client ID')
    df.dropna(subset=['Client.ID'], inplace=True)

    return df


def read_holiday_data(file_name, verbose=False):
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    print_full(df)
    print(df.dtypes)
    return df


def read_water_sensor_data(verbose=False):
    '''
    Downloads and reads water sensor data from the Chicago data
    portal. Downsamples the readings into the min, mean, and max
    for each day and for each sensor. Each day only has one row,
    with many columns (one column each per sensor per reading per
    type of down-sampling process)
    '''
    url = 'https://data.cityofchicago.org/api/views/qmqz-2xku/rows.csv?accessType=DOWNLOAD'
    water_sensors = pd.read_csv(url)
    url = 'https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD'
    sensor_locations = pd.read_csv(url)

    df = pd.merge(water_sensors, sensor_locations,
                  left_on='Beach Name', right_on='Sensor Name')

    df.drop(['Sensor Type', 'Location'], 1, inplace=True)

    # TODO: map sensor to beach ???

    df['Beach Name'] = df['Beach Name'].apply(lambda x: x[0:-6])

    df['Measurement Timestamp'] = pd.to_datetime(df['Measurement Timestamp'])
    df['Date'] = pd.DatetimeIndex(df['Measurement Timestamp']).normalize()
    df.drop(['Battery Life', 'Measurement Timestamp', 'Measurement Timestamp Label',
             'Measurement ID', 'Sensor Name'], axis=1, inplace=True)

    df_mins = df.groupby(['Beach Name', 'Date'], as_index=False).min()
    df_means = df.groupby(['Beach Name', 'Date'], as_index=False).mean()
    df_maxes = df.groupby(['Beach Name', 'Date'], as_index=False).max()

    cols = df_mins.columns.tolist()

    def rename_columns(cols, aggregation_type):
        cols = map(lambda x: x.replace(' ', '.'), cols)
        for i in range(2,7):
            cols[i] = cols[i] + '.' + aggregation_type
        return cols

    df_mins.columns = rename_columns(cols, 'Min')
    df_means.columns = rename_columns(cols, 'Mean')
    df_maxes.columns = rename_columns(cols, 'Max')

    df = pd.merge(df_mins, df_means, on=['Beach.Name', 'Date'])
    df = pd.merge(df, df_maxes, on=['Beach.Name', 'Date'])
    df.drop(['Latitude_x', 'Latitude_y', 'Longitude_x', 'Longitude_y'],
            axis=1, inplace=True)

    df = df.pivot(index='Date', columns='Beach.Name')
    df.columns = ['.'.join(col[::-1]).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.columns = ['Full_date'] + map(lambda x: x.replace(' ', '.'),
                                     df.columns.tolist()[1:])

    return df


def read_weather_station_data(verbose=False):
    '''
    Downloads and reads weather sensor data from the Chicago data
    portal. Downsamples the readings into the min, mean, and max
    for each day and for each sensor. Each day only has one row,
    with many columns (one column each per sensor per reading per
    type of down-sampling process)
    '''
    url = 'https://data.cityofchicago.org/api/views/k7hf-8y75/rows.csv?accessType=DOWNLOAD'
    weather_sensors = pd.read_csv(url)
    url = 'https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD'
    sensor_locations = pd.read_csv(url)

    weather_sensors.columns = map(lambda x: x.replace(' ', '.'),
                                  weather_sensors.columns.tolist())
    sensor_locations.columns = map(lambda x: x.replace(' ', '.'),
                                   sensor_locations.columns.tolist())
    sensor_locations.columns = ['Station.Name'] + sensor_locations.columns.tolist()[1:]

    df = pd.merge(weather_sensors, sensor_locations, on='Station.Name')

    df['Date'] = pd.DatetimeIndex(df['Measurement.Timestamp']).normalize()

    df.drop(['Measurement.Timestamp.Label', 'Measurement.Timestamp',
             'Sensor.Type', 'Location', 'Measurement.ID', 'Battery.Life'],
            axis=1, inplace=True)

    df_mins = df.groupby(['Station.Name', 'Date'], as_index=False).min()
    df_means = df.groupby(['Station.Name', 'Date'], as_index=False).mean()
    df_maxes = df.groupby(['Station.Name', 'Date'], as_index=False).max()

    cols = df_mins.columns.tolist()

    def rename_columns(cols, aggregation_type):
        cols = map(lambda x: x.replace(' ', '.'), cols)
        for i in range(2,15):
            cols[i] = cols[i] + '.' + aggregation_type
        return cols

    df_mins.columns = rename_columns(cols, 'Min')
    df_means.columns = rename_columns(cols, 'Mean')
    df_maxes.columns = rename_columns(cols, 'Max')

    df = pd.merge(df_mins, df_means, on=['Station.Name', 'Date'])
    df = pd.merge(df, df_maxes, on=['Station.Name', 'Date'])
    df.drop(['Latitude_x', 'Latitude_y', 'Longitude_x', 'Longitude_y'],
            axis=1, inplace=True)

    df = df.pivot(index='Date', columns='Station.Name')
    df.columns = ['.'.join(col[::-1]).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.columns = ['Full_date'] + map(lambda x: x.replace(' ', '.'),
                                     df.columns.tolist()[1:])

    return df


def print_full(x):
    '''
    Helper function to plot the *full* dataframe.
    '''
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def date_lookup(s, verbose=False):
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
            logging.debug('Non-regular date format "{0}"'.format(date))
            fmt = '%B %d (%p) %Y'
            dates[date] = pd.to_datetime(date,format=fmt)
    return s.apply(lambda v: dates[v])


def read_data(verbose=False):
    '''
    Read in the excel files for years 2006-2015 found in
    'data/ChicagoParkDistrict/raw/Standard 18 hr Testing'
    along with drekbeach data.

    Also reformats columns in accordance with the transformations
    found in analysis.R
    '''

    cpd_data_path = '../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    cpd_data_path = os.path.join(os.path.dirname(__file__), cpd_data_path)

    dfs = []

    for yr in range(2006,2015):
        dfs.append(split_sheets(cpd_data_path + str(yr) + ' Lab Results.xls', yr))
    dfs.append(split_sheets(cpd_data_path + '2015 Lab Results.xlsx', 2015))

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
                    if 'sample' in df.ix[i, col].lower():
                        logging.debug('Trying to cast "{0}" to numeric'.format(
                            df.ix[i, col]
                        ))
                    else:
                        logging.info('Trying to cast "{0}" to numeric'.format(
                            df.ix[i, col]
                        ))
                    df.ix[i, col] = float('nan')
        df[col] = df[col].astype('float64')

    # Massage dates, create weekday column
    df.insert(0, 'Full_date',
              df[['Date', 'Year']].apply(lambda x: ' '.join(x), axis=1))
    df['Full_date'] = date_lookup(df['Full_date'])
    days = ['Monday','Tuesday','Wednesday',
            'Thursday','Friday','Saturday','Sunday']
    df['Weekday'] = df['Full_date'].map(lambda x: days[x.dayofweek])
    # TODO: R code creates month/day columns too, do we need that?

    # Some header rows were duplicated
    df = df[df['Laboratory.ID'] != u'Laboratory ID']

    # Normalize the beach names
    df['Client.ID'] = df['Client.ID'].map(lambda x: x.strip())
    cleanbeachnames = pd.read_csv(cpd_data_path + 'cleanbeachnames.csv')
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
                  right_on=['Beach', 'Date'])
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

    # R code creates a calculated geometric mean column b/c it didn't
    # import the column correctly (it truncated the value). Pandas did
    # import correctly, so no need to create that.

    external_data_path = '../data/ExternalData/'
    external_data_path = os.path.join(os.path.dirname(__file__),
                                      external_data_path)

    holidaydata = read_holiday_data(external_data_path + 'Holidays.csv', verbose)
    # TODO: merge holiday data

    watersensordata = read_water_sensor_data(verbose)
    df = pd.merge(df, watersensordata, on='Full_date')

    weatherstationdata = read_weather_station_data(verbose)
    df = pd.merge(df, weatherstationdata, on='Full_date')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process beach data.')
    parser.add_argument('-o', '--outfile', nargs=1, type=str,
                        metavar='outfile', help='output CSV filename')
    parser.add_argument('-v', '--verbose', action='count')

    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    df = read_data(args.verbose)

    if args.outfile is not None:
        df.to_csv(args.outfile[0], index=False)
