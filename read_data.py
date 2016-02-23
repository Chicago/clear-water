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

# Added functions:

    # addColumnPriorData(df, colName , NdaysPrior):
        # Returns a new dataframe with an additional column of data N days Prior to current.

    # cleanUpBeaches(data):
        # Uses cleanbeachnames2.csv file for cleaning names. 
        # Also checks for duplicate readings at a given beach for a given day and takes the higher reading. 

    # read_data_simplified():
        # Simplifies data reading (no more need for split_sheets or date_lookup functions).
        # Also sorts Reading1 and Reading2 into lowReading and highReading columns.
        # Does minimal cleaning of data. 
        # No merging onto Drek or weather data.
    
    



def read_data_simplified():
    cpd_data_path = './data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    df =[]
    dfs =[]
    for yr in range(2002,2016):
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
                    discard_cols = set(dfx.columns) - set(['Year','Date','Client ID','Reading 1','Reading 2','Escherichia coli'])
                    if len(list(discard_cols))>0 :
                        dfx.drop(discard_cols, axis=1, inplace=True)
                    if len(dfx)>0 :
                        dfs.append(dfx)
                except KeyError:
                    continue
                    #print('Trouble processing :', sheet_name, yr)
                    #checked all trouble sheets and verified bad data
    df = pd.concat(dfs)

    # Do minimal processing of data
    df=df.rename(columns = {'Client ID':'Beach', 'Escherichia coli':'Ecoli_geomean', 'Reading 1':'Reading1'  ,'Reading 2': 'Reading2'})
    
    df['Reading1'] = df['Reading1'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Reading1'] = pd.to_numeric(df['Reading1'], errors='coerce')
    df['Reading2'] = df['Reading2'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Reading2'] = pd.to_numeric(df['Reading2'], errors='coerce')
    df['Ecoli_geomean'] = df['Ecoli_geomean'].map(lambda x: str(x).replace('<', '').replace('>', '') )
    df['Ecoli_geomean'] = pd.to_numeric(df['Ecoli_geomean'], errors='coerce')
    df['lowReading'] = df.Reading1 * (df.Reading1 < df.Reading2) + df.Reading2 * (df.Reading1 >= df.Reading2)
    df['highReading'] = df.Reading1 * (df.Reading1 >= df.Reading2) + df.Reading2 * (df.Reading1 < df.Reading2)
    df.drop(['Reading1','Reading2'], axis=1,inplace=True )

    df.insert(0, 'Full_date', df[['Date', 'Year']]
             .apply(lambda x: ' '.join(x), axis=1)
             .apply(lambda x: x.replace(' (PM)', '').replace(' PM', '') ))
    df.insert(0, 'Timestamp', pd.to_datetime(df['Full_date'],  errors='coerce') )
    df = df.dropna(axis=0, how='any', subset=['Timestamp','Full_date','Ecoli_geomean'])
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df.insert(0, 'Month', df['Timestamp'].dt.month.apply(lambda x: months[int(x)-1]) )
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df.insert(0, 'Weekday', df['Timestamp'].dt.dayofweek.apply(lambda x: days[int(x)]) )
    df.insert(0, 'DayofMonth', df['Timestamp'].dt.day)
    df.drop(['Date'], axis=1,inplace=True )
    
    df['Beach'] = df['Beach'].map(lambda x: x.strip())
    
    
    df = df[['Full_date','Timestamp','Beach','Year','Month','DayofMonth','Weekday','lowReading','highReading','Ecoli_geomean']]
    
    c = df.columns.tolist()
    c[c.index('Full_date')] = 'Date'
    df.columns = c
    
    #df = clean_up_beaches(df)
    return df



def clean_up_beaches(data):
    df = data.copy()
    cpd_data_path = './data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    cleanbeachnames = pd.read_csv(cpd_data_path + 'cleanbeachnames2.csv')
    cleanbeachnames = dict(zip(cleanbeachnames['Old'], cleanbeachnames['New']))
    df['Beach'] = df['Beach'].map(lambda x: cleanbeachnames[x])
    df = df.dropna(axis=0,  subset=['Beach'])
    df = df.reset_index()
    df['TempDate'] = df['Timestamp'].map(lambda x : str(x)[:10])
    df['Beach-Date'] = pd.Series(map((lambda x,y: str(x)+" : "+str(y)),df.Beach, df.TempDate)) 
    
    dupIndx=list(df.ix[df.duplicated(['Beach-Date'])].index)
    print('Colapsing {0} records'.format( len(dupIndx) ))
    df.drop(dupIndx,axis=0,inplace=True) # simply drop the first one. 
    df.drop(['index','TempDate','Beach-Date'], axis=1,inplace=True )

    return df   
    



def add_column_prior_data(df, colName , NdaysPrior):
    import datetime as dt
    dfc = df.copy()
    newColName = str(NdaysPrior) + '_day_prior_'+ colName 
    
    beaches = list(pd.Series(df['Beach'].unique()))
    days = pd.date_range(df['Timestamp'].min(), df['Timestamp'].max() , freq='d')
    df2 = dfc.set_index(['Beach','Timestamp'] )
    idx = pd.MultiIndex.from_product([beaches,days], names=['Beach','Timestamp'])
    full = df2.reindex(idx) # This ensures indexing only within beach
    previous = full.groupby(level=0)[colName].shift(NdaysPrior)
    previous.name = newColName 
    final = pd.concat([full,previous], axis=1)
    final.dropna(subset=[colName], inplace=True)
    
    dfn = final.reset_index()
    
    return dfn



  


    
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
            try :
                dates[date] = pd.to_datetime(date,format=fmt)
            except: 
                print('trouble processing Date: ', date)
    return s.apply(lambda v: dates[v])  


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
        'Date', 'Laboratory_ID', 'Beach', 'Reading1',
        'Reading2', 'Escherichia_coli', 'Units', 'Sample_Collection_Time'
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
            logging.debug('ignoring sheet "{0}" from {1}'.format(sheet_name, year))
            print('more than 30 columns for some sheet')
            continue

        if df.index.dtype == 'object':
            # If the first column does not have a label, then the excel
            # parsing engine will helpfully use the first column as
            # the index. This is *usually* helpful, but there are two
            # days when the first column is missing the typical label
            # of 'Laboratory ID'. In this case, peel that index off
            # and set its name.
            msg = '1st column in sheet "{0}" from {1} is missing title'.format(sheet_name, year)
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
    df.dropna(subset=['Beach'], inplace=True)

    return df


def read_holiday_data(file_name, verbose=False):
    df = pd.read_csv(file_name)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def read_water_sensor_data(verbose=False):
    '''
    Downloads and reads water sensor data from the Chicago data
    portal. Downsamples the readings into the min, mean, and max
    for each day and for each sensor. Each day only has one row,
    with many columns (one column each per sensor per reading per
    type of down-sampling process)
    '''
    #url = 'https://data.cityofchicago.org/api/views/qmqz-2xku/rows.csv?accessType=DOWNLOAD'
    #water_sensors = pd.read_csv(url)
    water_sensors = pd.read_csv('waterSensorDataDownloaded.csv')
    #url = 'https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD'
    #sensor_locations = pd.read_csv(url)
    sensor_locations = pd.read_csv('sensorLocationsDataDownloaded.csv')

    df = pd.merge(water_sensors, sensor_locations, left_on='Beach Name', right_on='Sensor Name')

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
    df_mins.drop(['Latitude','Longitude'],1,inplace=True)
    df_means.drop(['Latitude','Longitude'],1,inplace=True)
    df_maxes.drop(['Latitude','Longitude'],1,inplace=True)

    cols = df_mins.columns.tolist()

    def rename_columns(cols, aggregation_type):
        cols = list(map(lambda x: x.replace(' ', '_'), cols))
        for i in range(2,7):
            cols[i] = cols[i] + '_' + aggregation_type
        return cols

    df_mins.columns = rename_columns(cols, 'Min')
    df_means.columns = rename_columns(cols, 'Mean')
    df_maxes.columns = rename_columns(cols, 'Max')

    df = pd.merge(df_mins, df_means, on=['Beach_Name', 'Date'])
    df = pd.merge(df, df_maxes, on=['Beach_Name', 'Date'])

    df = df.pivot(index='Date', columns='Beach_Name')
    df.columns = ['.'.join(col[::-1]).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.columns = ['Full_date'] + list( map(lambda x: x.replace(' ', '_'), df.columns.tolist()[1:]))
    c = df.columns.tolist()
    c[c.index('Full_date')] = 'Date'
    df.columns = c

    return df


def read_weather_station_data(verbose=False):
    '''
    Downloads and reads weather sensor data from the Chicago data
    portal. Downsamples the readings into the min, mean, and max
    for each day and for each sensor. Each day only has one row,
    with many columns (one column each per sensor per reading per
    type of down-sampling process)
    '''
    #url = 'https://data.cityofchicago.org/api/views/k7hf-8y75/rows.csv?accessType=DOWNLOAD'
    #weather_sensors = pd.read_csv(url)
    weather_sensors = pd.read_csv('weatherSensorDataDownload.csv')
    #url = 'https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD'
    #sensor_locations = pd.read_csv(url)
    sensor_locations = pd.read_csv('sensorLocationsDataDownloaded.csv')

    weather_sensors.columns = map(lambda x: x.replace(' ', '_'),
                                  weather_sensors.columns.tolist())
    sensor_locations.columns = map(lambda x: x.replace(' ', '_'),
                                   sensor_locations.columns.tolist())
    sensor_locations.columns = ['Station_Name'] + sensor_locations.columns.tolist()[1:]

    df = pd.merge(weather_sensors, sensor_locations, on='Station_Name')

    df['Beach'] = df['Station_Name']

    df['Date'] = pd.DatetimeIndex(df['Measurement_Timestamp']).normalize()

    df.drop(['Measurement_Timestamp_Label', 'Measurement_Timestamp',
             'Sensor_Type', 'Location', 'Measurement_ID', 'Battery_Life','Station_Name'],
            axis=1, inplace=True)

    df_mins = df.groupby(['Beach', 'Date'], as_index=False).min()
    df_means = df.groupby(['Beach', 'Date'], as_index=False).mean()
    df_maxes = df.groupby(['Beach', 'Date'], as_index=False).max()

    cols = df_mins.columns.tolist()

    def rename_columns(cols, aggregation_type):
        cols = list(map(lambda x: x.replace(' ', '_'), cols))
        for i in range(2,15):
            cols[i] = cols[i] + '_' + aggregation_type
        return cols

    df_mins.columns = rename_columns(cols, 'Min')
    df_means.columns = rename_columns(cols, 'Mean')
    df_maxes.columns = rename_columns(cols, 'Max')


    df = pd.merge(df_mins, df_means, on=['Beach', 'Date'])
    df = pd.merge(df, df_maxes, on=['Beach', 'Date'])
    df.drop(['Latitude_x', 'Latitude_y', 'Longitude_x', 'Longitude_y'], axis=1, inplace=True)

    df = df.pivot(index='Date', columns='Beach')
    df.columns = ['.'.join(col[::-1]).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    df.columns = ['Full_date'] + list( map(lambda x: x.replace(' ', '_'), df.columns.tolist()[1:]))
    c = df.columns.tolist()
    c[c.index('Full_date')] = 'Date'
    df.columns = c

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





def read_data(verbose=False):
    '''
    Read in the excel files for years 2006-2015 found in
    'data/ChicagoParkDistrict/raw/Standard 18 hr Testing'
    along with drekbeach data.

    Also reformats columns in accordance with the transformations
    found in analysis.R
    '''

    cpd_data_path = './data/ChicagoParkDistrict/raw/Standard 18 hr Testing/'
    #cpd_data_path = os.path.join(os.path.dirname(__file__), cpd_data_path)

    dfs = []

    for yr in range(2002,2015):
        dfs.append(split_sheets(cpd_data_path + str(yr) + ' Lab Results.xls', yr))
    dfs.append(split_sheets(cpd_data_path + '2015 Lab Results.xlsx', 2015))

    df = pd.concat(dfs)

    # Need to reset the index to deal with the repeated concatenations
    df.index = range(0, len(df.index))

    # Some records are of the form <1 or >2440
    # Remove the operator and treat the remaining string as the value.
    # Also convert string to float, if possible
    for col in ['Reading1', 'Reading2', 'Escherichia_coli']:
        for i, val in enumerate(df[col].tolist()):
            if isinstance(val, (str,bytes)):
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
    df.insert(0, 'Full_date', df[['Date', 'Year']].apply(lambda x: ' '.join(x), axis=1).apply(lambda x: x.replace(' (PM)', '') ))
    df['Full_date'] = date_lookup(df['Full_date'])
    df.insert(0, 'Timestamp', pd.to_datetime(df['Full_date'],  errors='coerce') )
    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    df.insert(0, 'Month', df['Timestamp'].dt.month.apply(lambda x: months[int(x)-1]) )
    days=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    df.insert(0, 'Weekday', df['Timestamp'].dt.dayofweek.apply(lambda x: days[int(x)]) )
    df.drop(['Date','Timestamp'], axis=1,inplace=True )

    # Some header rows were duplicated
    df = df[df['Laboratory_ID'] != u'Laboratory ID']
    # Normalize the beach names
    df['Beach'] = df['Beach'].map(lambda x: x.strip())
    cleanbeachnames = pd.read_csv(cpd_data_path + 'cleanbeachnames.csv')
    cleanbeachnames = dict(zip(cleanbeachnames['Old'], cleanbeachnames['New']))
    # There is one observation that does not have a beach name in the
    # Beach column, remove it.
    df = df[df['Beach'].map(lambda x: x in cleanbeachnames)]
    df['Beach'] = df['Beach'].map(lambda x: cleanbeachnames[x])

    df['Beach-Date'] = list(map((lambda x,y: str(x)+"-"+str(y)[:10]),df.Beach, df.Full_date))
    
    
    # Don't get drek data merged yet.
    # Read in drek beach data 
    # drek_data_path = './data/DrekBeach/'
    # drekdata = pd.read_csv(drek_data_path + 'daily_summaries_drekb.csv')
    # drekdata.columns = ['Beach', 'Full_date', 'Drek_Reading','Drek_Prediction', 'Drek_Worst_Swim_Status']
    # drekdata['Full_date'] = date_lookup(drekdata['Full_date'])
    # drekdata['Beach'] = drekdata['Beach'].map(lambda x: x.strip())
    # drekdata['Beach'] = drekdata['Beach'].map(lambda x: cleanbeachnames[x])
    # df = pd.merge(df, drekdata, how='outer', on= ['Beach', 'Full_date'])
    
    c = df.columns.tolist()
    c[c.index('Full_date')] = 'Date'
    df.columns = c

        
    # get rid of some useless columns
    #df.drop(['Laboratory_ID','Units','Sample_Collection_Time','Drek_Worst_Swim_Status'], axis=1,inplace=True )
    df.drop(['Laboratory_ID','Units','Sample_Collection_Time'], axis=1,inplace=True )
    
    # There was an anamolous reading, the max possible value from the test
    # is around 2420, but one reading was 6488.
    # We need to do the ~(reading 1 > 2500 | reading 2 > 2500) instead of
    # (reading 1 < 2500 & reading 2 < 2500) since the latter returns
    # False if there is a NaN.
    df = df[~((df['Reading1'] > 2500) | (df['Reading2'] > 2500))]

    # R code creates a calculated geometric mean column b/c it didn't
    # import the column correctly (it truncated the value). Pandas did
    # import correctly, so no need to create that.

    #external_data_path = './data/ExternalData/'
    #external_data_path = os.path.join(os.path.dirname(__file__),
    #                                  external_data_path)

    #holidaydata = read_holiday_data(external_data_path + 'Holidays.csv', verbose)
    # TODO: merge holiday data

    #watersensordata = read_water_sensor_data(verbose)
    #df = pd.merge(df, watersensordata, on='Date', how='outer')

    #weatherstationdata = read_weather_station_data(verbose)
    #df = pd.merge(df, weatherstationdata, on='Date', how='outer')

    # TODO: discuss this
    #df.set_index('Date', drop=True, inplace=True)

    df['actual_elevated'] = (df['Escherichia_coli']>=235).astype(int)
    #df['predicted_elevated'] = (df['Drek_Prediction']>=235).astype(int)

    df['smallReading'] = df.Reading1 * (df.Reading1 < df.Reading2) + df.Reading2 * (df.Reading1 >= df.Reading2)
    df['largeReading'] = df.Reading1 * (df.Reading1 >= df.Reading2) + df.Reading2 * (df.Reading1 < df.Reading2)

    
    #df = df.ix[pd.notnull(df['Escherichia_coli'])].reset_index()
    #df.drop(['index'], axis=1, inplace=True)
    df = df.ix[pd.notnull(df['Beach'])].reset_index()
    df.drop(['index'], axis=1, inplace=True)

    # get levels of ecoli from yesterday and day before yesterday
    import datetime as dt
    
    #df = getYesterdaysData(df,'Escherichia_coli')
    #df = getYesterdaysData(df,'smallReading')
    #df = getYesterdaysData(df,'largeReading')
    

    # temp = df.ix[:,['Date','Beach','Escherichia_coli']].reset_index()
    # temp['DateTwoDaysAhead']= temp['Date'] + dt.timedelta(days=2)
    # temp['DayBeforeYesterdayEcoliGeom'] = temp['Escherichia_coli']
    # temp.drop(['index','Date','Escherichia_coli'], axis=1, inplace=True)
    # df = pd.merge(df, temp, left_on=['Beach', 'Date'], right_on=['Beach', 'DateTwoDaysAhead'], how='left')
    # df.drop(['DateTwoDaysAhead'], 1, inplace=True)




    return df





def getYesterdaysData(df,colName):
    import datetime as dt
    newColName = 'Yesterday_'+ colName
    temp = df.ix[:,['Date','Beach',colName]].reset_index()
    temp['DateTomorrow']= temp['Date'] + dt.timedelta(days=1)
    temp[newColName] = temp[colName]
    temp.drop(['index','Date',colName], axis=1, inplace=True)
    df = pd.merge(df, temp, left_on=['Beach', 'Date'], right_on=['Beach', 'DateTomorrow'], how='left')
    df.drop(['DateTomorrow'], 1, inplace=True)
    return df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process beach data.')
    parser.add_argument('-o', '--outfile', nargs=1, type=str,
                        metavar='outfile', help='output CSV filename')
    parser.add_argument('-v', '--verbose', action='count')

    args = parser.parse_args()

    
    if args.verbose == None:
        args.verbose = 0

    if args.verbose >= 2:
       logging.basicConfig(level=logging.DEBUG)
    elif args.verbose == 1:
       logging.basicConfig(level=logging.INFO)
    else:
       logging.basicConfig(level=logging.WARNING)

    df = read_data(args.verbose)

    if args.outfile is not None:
        df.to_csv(args.outfile[0], index=False)
