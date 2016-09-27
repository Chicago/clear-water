from __future__ import print_function
import pandas as pd
import numpy as np
import logging
import argparse
import six
import datetime as dt

'''
This file reads in data related E. coli levels
in Chicago beaches. It is based on the files
analysis.R and split_sheets.R, and is written
such that the dataframe loaded here will match
the R dataframe code exactly.
'''


def read_data_simplified():
    '''
    Simple function to read in the data from the excel files.

    Returns a dataframe that has only been minimally cleaned.
    '''
    cpd_data_path = '../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    df =[]
    dfs =[]
    for yr in range(2002,2015):
        if yr != 2004:
            file_name = cpd_data_path + str(yr) + ' Lab Results.xls'
            xls = pd.ExcelFile(file_name)
            for i, sheet_name in enumerate(xls.sheet_names):
                dfx = xls.parse(sheet_name)
                if (yr == 2002) or (yr == 2003):
                    if len(dfx.columns)==2:
                        dfx.columns = ['Client ID','Escherichia coli']
                        dfx["Escherichia coli"]= dfx["Escherichia coli"].astype(str)
                    elif len(dfx.columns)==3:
                        dfx.columns = ['Client ID','Escherichia coli','other']
                        dfx["Escherichia coli"]= dfx["Escherichia coli"].astype(str)
                try :
                    dfx = dfx.dropna(axis=0, how='any', subset=['Escherichia coli','Client ID'])
                    dfx.insert(0, u'Date', sheet_name)
                    dfx.insert(0, u'Year', str(yr))
                    discard_cols = set(dfx.columns) - set(['Year','Date','Client ID','Reading 1',
                                                           'Reading 2','Escherichia coli'])
                    if len(list(discard_cols))>0 :
                        dfx.drop(discard_cols, axis=1, inplace=True)
                    if len(dfx)>0 :
                        dfs.append(dfx)
                except KeyError:
                    continue
                    # print('Trouble processing :', sheet_name, yr)
                    # checked all trouble sheets and verified bad data
    df = pd.concat(dfs)

    # Do minimal processing of data
    df=df.rename(columns = {'Client ID':'Beach', 'Escherichia coli':'Ecoli_geomean',
                            'Reading 1':'Reading1'  ,'Reading 2': 'Reading2'})

    df['Reading1'] = df['Reading1'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Reading1'] = pd.to_numeric(df['Reading1'], errors='coerce')
    df['Reading2'] = df['Reading2'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Reading2'] = pd.to_numeric(df['Reading2'], errors='coerce')
    df['Ecoli_geomean'] = df['Ecoli_geomean'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Ecoli_geomean'] = pd.to_numeric(df['Ecoli_geomean'], errors='coerce')
    df['lowReading'] = df.Reading1 * (df.Reading1 < df.Reading2) + df.Reading2 * (df.Reading1 >= df.Reading2)
    df['highReading'] = df.Reading1 * (df.Reading1 >= df.Reading2) + df.Reading2 * (df.Reading1 < df.Reading2)
    df.drop(['Reading1','Reading2'], axis=1,inplace=True )

    df.insert(0, 'Full_date', df[['Date', 'Year']].apply(lambda x: ' '.join(x), axis=1).apply(lambda x: x.replace(' (PM)', '') ))
    df.insert(0, 'Timestamp', pd.to_datetime(df['Full_date'],  errors='coerce') )
    df = df.dropna(axis=0, how='any', subset=['Timestamp','Full_date','Ecoli_geomean'])
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df.insert(0, 'Month', df['Timestamp'].dt.month.apply(lambda x: months[int(x)-1]) )
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df.insert(0, 'Weekday', df['Timestamp'].dt.dayofweek.apply(lambda x: days[int(x)]) )
    df.insert(0, 'DayofMonth', df['Timestamp'].dt.day)
    df.drop(['Date'], axis=1,inplace=True )

    df['Beach'] = df['Beach'].map(lambda x: x.strip())

    df = df[['Full_date','Timestamp','Beach','Year','Month','DayofMonth',
             'Weekday','lowReading','highReading','Ecoli_geomean']]
    df = df.reset_index()
    df.drop(['index'], axis=1, inplace=True)

    return df

def group_beaches_geographically(data, beach_names_column='Client.ID', verbose=False):
    '''
    Creates geographical categorizations for the beaches.

    The indicators are any of six groups and a north indicator (1 if North of the pier, 0 if South).
    '''
    df = data.copy()

    # Mid North
    group_1 = ['Montrose','Montrose Dog','Foster','Osterman','Leone','Thorndale']

    # Rogers Park Area
    group_2 = ['Juneway','Rogers','Howard','Marion','Leone','Pratt','Columbia','North Shore','Hartigan','Albion',
            'Marion Mahoney Griffin','Loyola','Margaret T Burroughs','Lane','Jarvis']
    # Near North Area
    group_3 = ['Oak Street','Ohio','North Ave','North Avenue']

    # Near South
    group_4 = ['12th','31st']

    # Mid South

    group_5 = ['57th','63rd','41st','39th','Oakwood','67th','South Shore','Rainbow', 'Humbolt']

    # Calumet is way off to the side.
    group_6 = ['Calumet']

    north_group = group_1 + group_2 + group_3
    south_group = group_4 + group_5 + group_6
    all_groups = [group_1, group_2, group_3, group_4, group_5, group_6]

    for ind in range(len(all_groups)):
        var_name = 'flag_geographic_group_' + str(ind+1)
        df[var_name] = df[beach_names_column].map(lambda x: beach_grouping(x,all_groups[ind]))

    df['flag_geographically_a_north_beach'] = df[beach_names_column].map(lambda x: beach_grouping(x,north_group))

    # also want to add single columns, but that is harder.
    df['categorical_beach_grouping'] = df[beach_names_column].map(lambda x: single_grouping(x,all_groups))

    return df

def single_grouping(beach_name, groups):
    '''Takes list of lists of strings, and sees if thed given beach_name matches any of those.  It then reports which of the initial lists it was in.  If none are found, it returns an N+1th category.

    Inputs
    ------
    beach_name: string
    groups: list of length N, of lists of strings
    '''

    for beach_group in range(len(groups)):
        if beach_name in groups[beach_group]:
            return 'beach_in_grouping_' + str(beach_group + 1)
    #print('no category: ' + str(beach_name)) #line used in debugging.
    return 'beach_not_in_any_grouping'

def beach_grouping(beach_name, grouping):
    '''Simple function that returns 1 if the beach name is in the applied list.

    Input
    -----
    beach_name: string
    grouping: list of strings

    '''

    if beach_name in grouping:
        return int(1)
    else:
        return int(0)

def clean_up_beaches(data, beach_names_column='Beach', verbose=False):
    '''
    Merge beach names to prevent duplicate readings.

    Parameters
    ==========
    data               : A dataframe containing a column of beach names
    beach_names_column : The name of the column containing the beach names
    verbose            : Print information if True

    Output
    ======
    df : Dataframe with cleaned beach names.
    '''
    df = data.copy()
    cpd_data_path = '../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    cleanbeachnames = pd.read_csv(cpd_data_path + 'cleanbeachnames2.csv')
    cleanbeachnames = dict(zip(cleanbeachnames['Old'], cleanbeachnames['New']))
    df[beach_names_column] = df[beach_names_column].map(lambda x: cleanbeachnames[x])
    df = df.dropna(axis=0,  subset=[beach_names_column])
    df['Beach-Date'] = list(map((lambda x,y: str(x)+" : "+str(y)),
                                df[beach_names_column], df.Full_date))

    dupIndx=list(df.ix[df.duplicated(['Beach-Date'])].index)
    if verbose:
        print('Colapsing {0} records by taking highest reading'.format( len(dupIndx) ))

    # assuming that the name-date conflict is for at most two records
    for idx in range(len(dupIndx)):
        BD = df.ix[dupIndx[idx],['Beach-Date']]
        TmpIndx=list(df.ix[df['Beach-Date']==BD[0]].index)
        TmpVal0=df.ix[TmpIndx[0],['Ecoli_geomean']]
        TmpVal1=df.ix[TmpIndx[1],['Ecoli_geomean']]
        if TmpVal0[0] > TmpVal1[0] :
            df.drop(TmpIndx[0],axis=0,inplace=True)
        else:
            df.drop(TmpIndx[1],axis=0,inplace=True)

    return df


def add_columns_each_beach(df, beach_col_name='Beach', timestamp_col_name='Timestamp', ecoli_col_name='Ecoli_geomean'):
    '''
    Adds a column containing the previous day's Ecoli_geomean reading at each beach.
    Also calls add_column_prior_data to add on previous 7 days' of readings.
    Fills in missing data to 'previous_reading' using mean of previous 7 days' of readings.

    Parameters
    ==========
    df       : Dataframe of data.

    Keyword Arguments
    =================
    beach_col_name='Beach'
        The name of the column containing the beach names in df.
    timestamp_col_name='Timestamp'
        The name of the column containing the timestamp in df.

    Output
    ======
    dfc : Dataframe with the previous data merged onto (a copy of) the input dataframe.
          Adds on group_prior_mean based on which of 4 categories the beach is in 

    '''
    dfc = df.copy()
    dfc = add_column_prior_data(dfc, ecoli_col_name, [1,2,3,4,5,6,7] , beach_col_name=beach_col_name , timestamp_col_name=timestamp_col_name)
    col_list = [ '1_day_prior_'+ecoli_col_name , '2_day_prior_'+ecoli_col_name , '3_day_prior_'+ecoli_col_name,
                 '4_day_prior_'+ecoli_col_name , '5_day_prior_'+ecoli_col_name , '6_day_prior_'+ecoli_col_name,
                 '7_day_prior_'+ecoli_col_name]
    temp = dfc[col_list].copy()
    dfc['7_day_moving_avg'] = temp.mean(axis=1)
    dfc['previous_reading'] =  pd.Series(map((  lambda x,y: y if np.isnan(x) else x  ), dfc['1_day_prior_'+ecoli_col_name], dfc['7_day_moving_avg'])  )
    beaches = list(pd.Series(dfc[beach_col_name].value_counts().index))
    for j in range(len(beaches)):
        beach2 = dfc.ix[dfc[beach_col_name]==beaches[j],[timestamp_col_name,beach_col_name,'previous_reading']].reset_index()
        c = beach2.columns.tolist()
        c[c.index('previous_reading')] = beaches[j] + '_previous'
        beach2.columns = c
        beach2.drop(['index',beach_col_name], axis=1, inplace=True)
        dfc = pd.merge(dfc, beach2, on=timestamp_col_name, how='left')
        medianVal = dfc.loc[dfc[beach_col_name]==beaches[j], ecoli_col_name].median()
        dfc.loc[ np.isnan(dfc[beaches[j]+'_previous']) , [beaches[j]+'_previous']] = medianVal

    near_north  = ['Montrose_previous','Foster_previous','Osterman_previous']
    far_north   = ['Albion_previous','Jarvis_previous', 'Rogers_previous', 'Juneway_previous','Leone_previous','Howard_previous']
    city_center = ['Ohio_previous','Oak Street_previous','12th_previous','North Avenue_previous', '31st_previous']
    south_side  = ['South Shore_previous','63rd_previous', 'Rainbow_previous','Calumet_previous', '57th_previous','39th_previous']

    dfc['near_north_prior_mean'] = dfc[near_north].mean(axis=1)
    dfc['far_north_prior_mean'] = dfc[far_north].mean(axis=1)
    dfc['city_center_prior_mean'] = dfc[city_center].mean(axis=1)
    dfc['south_side_prior_mean'] = dfc[south_side].mean(axis=1)

    dfc['group_prior_mean'] = dfc['near_north_prior_mean']
    dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_2','group_prior_mean'] = dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_2','far_north_prior_mean']  
    dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_3','group_prior_mean'] = dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_3','city_center_prior_mean']  
    dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_4','group_prior_mean'] = dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_4','city_center_prior_mean']
    dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_5','group_prior_mean'] = dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_5','south_side_prior_mean']  
    dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_6','group_prior_mean'] = dfc.loc[dfc['categorical_beach_grouping']=='beach_in_grouping_6','south_side_prior_mean'] 
    
    return dfc





def add_column_prior_data(df, colnames, ns, beach_col_name='Beach', timestamp_col_name='Timestamp'):
    '''
    Adds data from previous days to the dataframe.

    Parameters
    ==========
    df       : Dataframe of data.
    colnames : Can be a string, integer, or list-like of strings and integers.
               If a list-like type is specified, then prior data will be added
               for each specified column.
    ns       : Can be an integer, or list-like of integers. This specifies how
               many days back the prior data should be added from. For example,
               if ns is 1, then data from the previous day will be added (if the
               previous day is unavailable, then a nan is inserted).
               If a list-like type is specified, then for each n in ns,
               the data from n days back is added.

    Keyword Arguments
    =================
    beach_col_name='Beach'
        The name of the column containing the beach names in df.
    timestamp_col_name='Timestamp'
        The name of the column containing the timestamp in df.

    Output
    ======
    df : Dataframe with the previous data merged onto (a copy of) the input dataframe.

    Example
    =======
    >>> import read_data as rd
    >>> df = rd.read_data(read_weather_station=False, read_water_sensor=False)
    >>> df2 = rd.add_column_prior_data(df, 'Escherichia.coli', 1,
    >>>                                beach_col_name='Client.ID', timestamp_col_name='Full_date')
    >>> df3 = rd.add_column_prior_data(df, 'Escherichia.coli', [1,2,3,4,5,6,7],
    >>>                                beach_col_name='Client.ID', timestamp_col_name='Full_date')
    >>> df4 = rd.add_column_prior_data(df, ['summary', 'icon'], [1, 2],
    >>>                                beach_col_name='Client.ID', timestamp_col_name='Full_date')
    '''
    if not hasattr(colnames, '__getitem__') or isinstance(colnames, six.string_types):
        colnames = [colnames]
    if not hasattr(ns, '__getitem__'):
        ns = [ns]

    if not colnames and ns:
        return df.copy()

    dfc = df.copy()
    for colname in colnames:
        for n in ns:
            new_col_name = str(n) + '_day_prior_'+ colname

            beaches = list(pd.Series(df[beach_col_name].unique()))
            days = pd.date_range(df[timestamp_col_name].min(), df[timestamp_col_name].max() , freq='d')
            df2 = dfc.set_index([beach_col_name,timestamp_col_name] )
            idx = pd.MultiIndex.from_product([beaches,days], names=[beach_col_name,timestamp_col_name])
            full = df2.reindex(idx) # This ensures indexing only within beach
            previous = full.groupby(level=0)[colname].shift(n)
            previous.name = new_col_name
            final = pd.concat([full,previous], axis=1)

            dfn = final.reset_index()
            df = dfn
            dfc = df.copy()

    return dfn.copy()


def process_hourly_data(df, hours_offset=0,
                        beach_col_name='Client.ID',
                        timestamp_col_name='Full_date'):
    '''
    Pivot hourly data into daily data, optionally shifting the hours that
    get mapped to a given date.

    For example, you can choose to have a "day" be defined as 5am to 5am, so
    that, e.g., all hours from 6/1/2015 05:00 through 6/2/2015 05:00 inclusive
    are pivoted onto the day 6/1/2015.

    Parameters
    ----------
    df                 : Dataframe to pivot. Should have hourly data.
    hours_offset       : The hour offset. All hours ranging from
                         ((midnight on day D) + hours_offset) through
                         ((midnight on day (D + 1)) + hours_offset) will
                         be pivoted onto day D.
    beach_col_name     : Name of the column containing the beach names.
    timestamp_col_name : Name of the column containing the timestamps.
                         Note that this column is expected to be
                         pandas.tslib.Timestamp.

    Returns
    -------
    df : Pivoted dataframe.

    Example
    --------
    >>> import read_data as rd
    >>> df = rd.read_forecast_data('../data/ExternalData/forecastio_hourly_weather.csv')
    >>> df2 = rd.process_hourly_data(df)
    >>> df3 = rd.process_hourly_data(df, 5)
    >>> df4 = rd.process_hourly_data(df, -5)
    '''
    df = df.copy()
    df[timestamp_col_name] = df[timestamp_col_name].map(
        lambda x: x - dt.timedelta(hours=hours_offset)
    )
    df['hour_of_day'] = df[timestamp_col_name].map(
        lambda x: x.hour + hours_offset
    )
    df[timestamp_col_name] = pd.to_datetime(df[timestamp_col_name].map(
        lambda x: x.date().isoformat()
    ))

    df = df.pivot_table(index=[beach_col_name, timestamp_col_name],
                        columns='hour_of_day')
    df.columns = [x[0] + '_hour_' + str(x[1]) for x in df.columns]
    df = df.reset_index()
    return df


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


def read_holiday_data(file_name):
    '''
    Reads in holiday CSV file.
    '''
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    df.columns = ['Full_date', 'Holiday']
    return df


def days_since_holiday(df):
    '''
    Creates a column that describes the number of days since last
    summer holiday, which could have happened last year resulting
    in numbers 300 and above.
    '''
    df['Holiday.Flag'] = ~df['Holiday'].isnull()

    holiday_dates = df.ix[df['Holiday.Flag'], 'Full_date'].unique()
    holiday_dates = pd.Series(holiday_dates)

    def day_count(d):
        less_than_date = holiday_dates.map(lambda x: x <= d)
        if not less_than_date.any():
            return float('nan')
        holiday_date = holiday_dates[less_than_date[less_than_date].index[-1]]
        delta = d - holiday_date
        return delta.days

    df['Days.Since.Last.Holiday'] = df['Full_date'].map(day_count)

    return df


def read_forecast_data(filename):
    '''
    Read in forecast.io historical weather data.
    '''

    df = pd.read_csv(filename)
    df = df.drop_duplicates()
    cols = df.columns.tolist()
    cols[cols.index('beach')] = 'Client.ID'
    cols[cols.index('time')] = 'Full_date'
    df.columns = cols
    df['Full_date'] = pd.to_datetime(df['Full_date'])

    return df

def convert_UNIX_times(df, column_list = ['sunriseTime','sunsetTime','temperatureMinTime','temperatureMaxTime','apparentTemperatureMinTime','apparentTemperatureMaxTime']):
	'''
	Turns UNIX times (seconds elapsed since Jan 1, 1970)  into a
	four-digit float best interpreted as 'hours since prior midnight.'
	This is done by taking the UNIX time in seconds, subtracting
	away the seconds equivalent of Full_date, and then subtracting
	5 hours to account for a conversion between GMT and Central Time

	The list of columns provided as a default to the column_list
	argument should be the only ones that need conversion
	'''
	for col in column_list:
		df[col] = [round(float(val.components[1] + val.components[2] * 1.0/60 + val.components[3]*1.0/3600),4) for val in\
					 pd.to_datetime(df[col], unit = 's') - pd.Timedelta(hours = 5) - pd.to_datetime(df.Full_date)]

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
        cols = [x.replace(' ', '.') for x in cols]
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
    df.columns = ['Full_date'] + [x.replace(' ', '.') for x in df.columns.tolist()[1:]]

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
        cols = [x.replace(' ', '.') for x in cols]
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
    df.columns = ['Full_date'] + [x.replace(' ', '.') for x in df.columns.tolist()[1:]]

    return df


def read_locations(file_name, verbose=False):
    locations = pd.read_csv(file_name)
    return locations


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
    for date, parsed in dates.items():
        if type(parsed) is not pd.tslib.Timestamp:
            logging.debug('Non-regular date format "{0}"'.format(date))
            fmt = '%B %d (%p) %Y'
            dates[date] = pd.to_datetime(date,format=fmt)
    return s.apply(lambda v: dates[v])


def read_data(verbose=False, read_drek=True, read_holiday=True, read_weather_station=True,
              read_water_sensor=True, read_daily_forecast=True, read_hourly_forecast=True,
        group_beaches=True):
    '''
    Read in the excel files for years 2006-2015 found in
    'data/ChicagoParkDistrict/raw/Standard 18 hr Testing'
    along with drekbeach data.

    Also reformats columns in accordance with the transformations
    found in analysis.R
    '''

    cpd_data_path = '../data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'

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
            if isinstance(val, six.string_types):
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
    cleanbeachnamesdf = pd.read_csv(cpd_data_path + 'cleanbeachnames.csv')
    cleanbeachnames = dict(zip(cleanbeachnamesdf['Old'], cleanbeachnamesdf['Short_Names']))
    # There is one observation that does not have a beach name in the
    # Client.ID column, remove it.
    df = df[df['Client.ID'].map(lambda x: x in cleanbeachnames)]
    df['Client.ID'] = df['Client.ID'].map(lambda x: cleanbeachnames[x])

    if read_drek:
        # Read in drek beach data
        drek_data_path = '../data/DrekBeach/'
        drekdata = pd.read_csv(drek_data_path + 'daily_summaries_drekb.csv')
        drekdata.columns = ['Beach', 'Date', 'Drek_Reading',
                            'Drek_Prediction', 'Drek_Worst_Swim_Status']
        drekdata['Date'] = date_lookup(drekdata['Date'])
        drekdata['Beach'] = drekdata['Beach'].map(lambda x: x.strip())
        drekdata['Beach'] = drekdata['Beach'].map(lambda x: cleanbeachnames[x])
        # Merge the data
        df = pd.merge(df, drekdata, how='outer',
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

    df = df.sort_values(by=['Full_date', 'Client.ID'])

    external_data_path = '../data/ExternalData/'

    if read_holiday:
        holidaydata = read_holiday_data(external_data_path + 'Holidays.csv')
        df = pd.merge(df, holidaydata, on='Full_date', how='outer')
        df = days_since_holiday(df)
        df['Holiday_Recently'] = (df['Days.Since.Last.Holiday']<3)

    if group_beaches:
        df = group_beaches_geographically(df)

    if read_daily_forecast:
        beach_names_new_to_short = dict(zip(cleanbeachnamesdf['New'],
                                            cleanbeachnamesdf['Short_Names']))

        forecast_daily = read_forecast_data(external_data_path + 'forecastio_daily_weather.csv')
        forecast_daily['Client.ID'] = forecast_daily['Client.ID'].map(
            lambda x: beach_names_new_to_short[x]
        )
        df = pd.merge(df, forecast_daily, on=['Full_date', 'Client.ID'])
        df = convert_UNIX_times(df, column_list = ['sunriseTime','sunsetTime','temperatureMinTime','temperatureMaxTime','apparentTemperatureMinTime','apparentTemperatureMaxTime'])
        # create useful interaction variables combining wind speed and direction
        df['windVectorX'] = np.cos(np.pi*2*df['windBearing']/360)*df['windSpeed']
        df['windVectorY'] = np.sin(np.pi*2*df['windBearing']/360)*df['windSpeed'] 

    if read_hourly_forecast:
        beach_names_new_to_short = dict(zip(cleanbeachnamesdf['New'],
                                            cleanbeachnamesdf['Short_Names']))
        forecast_hourly = read_forecast_data(external_data_path + 'forecastio_hourly_weather.csv')
        forecast_hourly['Client.ID'] = forecast_hourly['Client.ID'].map(
            lambda x: beach_names_new_to_short[x]
        )
        forecast_hourly = process_hourly_data(forecast_hourly, hours_offset=-19)
        df = pd.merge(df, forecast_hourly, on=['Full_date', 'Client.ID'])
        # create useful interaction variables combining wind speed and direction
        for hr in range(-19,5):
            bearing_col = 'windBearing_hour_' + str(hr)
            speed_col = 'windSpeed_hour_' +str(hr)
            vectorX_col = 'windVectorX_hour_'+str(hr)
            df[vectorX_col] = np.cos(np.pi*2*df[bearing_col]/360)*df[speed_col]


    if read_water_sensor:
        watersensordata = read_water_sensor_data()
        df = pd.merge(df, watersensordata, on='Full_date', how='outer')

    if read_weather_station:
        weatherstationdata = read_weather_station_data()
        df = pd.merge(df, weatherstationdata, on='Full_date', how='outer')

    # Final cleaning, drop rows with no beach name, remove duplicate beach-days
    df = df.dropna(subset=['Client.ID'])
    df.drop_duplicates(subset=['Client.ID', 'Full_date'], keep='first', inplace=True)

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process beach data.')
    parser.add_argument('-o', '--outfile', nargs=1, type=str,
                        metavar='outfile', help='output CSV filename')
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument('-t', '--test', action='count', help='run tests', default=0)

    args = parser.parse_args()

    if args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose == 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    if args.outfile is not None:
        df = read_data(args.verbose)
        df.to_csv(args.outfile[0], index=False)


    if args.test:
        df = read_data(args.verbose)
        print('read_data() returns dataframe with size {0} x {1}'.format(
            df.shape[0], df.shape[1]
        ))
        df2 = add_column_prior_data(df, 'Escherichia.coli', 1,
                                    beach_col_name='Client.ID',
                                    timestamp_col_name='Full_date')
        print('Adding a single prior column results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))
        df2 = add_column_prior_data(df, 'Escherichia.coli', range(4),
                                    beach_col_name='Client.ID',
                                    timestamp_col_name='Full_date')
        print('Adding a four prior days results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))
        df2 = add_column_prior_data(df, ['Escherichia.coli', 'icon'], range(4),
                                    beach_col_name='Client.ID',
                                    timestamp_col_name='Full_date')
        print('Adding a four prior days of two columns results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))

        df = read_data_simplified()
        print('read_data_simplified() returns dataframe with size {0} x {1}'.format(
            df.shape[0], df.shape[1]
        ))
        df = clean_up_beaches(df)
        print('After cleaning with clean_up_beaches, we have a dataframe with size {0} x {1}'.format(
            df.shape[0], df.shape[1]
        ))
        df2 = add_column_prior_data(df, 'Ecoli_geomean', 1)
        print('Adding a single prior column results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))
        df2 = add_column_prior_data(df, 'Ecoli_geomean', range(4))
        print('Adding a four prior days results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))
        df2 = add_column_prior_data(df, ['Ecoli_geomean', 'lowReading'], range(4))
        print('Adding a four prior days of two columns results in a dataframe with size {0} x {1}'.format(
            df2.shape[0], df2.shape[1]
        ))
