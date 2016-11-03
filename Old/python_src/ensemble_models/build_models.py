import numpy as np
import pandas as pd
import read_data as rd
import argparse
import os
import time
import sklearn
from sklearn.externals import joblib
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor




def prepare_data(df=None):
    '''
    Preps the data to be used in the model. Right now, the code itself must
    be modified to tweak which columns are included in what way.
    Parameters
    ----------
    df : Dataframe to use. If not specified, the dataframe is loaded automatically.
    Returns
    -------
    predictors : NxM DataFrame of the predictors for the classification problem.
    meta_info  : Nx6 DataFrame containing the columns 'Escherichia.coli' and
                 'Full_date', to be used, e.g., for leave-one-year-out cross
                 validation and creating the true class labels (elevated vs.
                 not elevated E. coli levels). The columns 'Client.ID','BEACH',
                 'Drek_Prediction'and 'Weekday' are also returned.
    '''

    # Meta columns are not used as predictors
    meta_columns = ['Client.ID','BEACH','Full_date','Escherichia.coli',
                    'Drek_Prediction','Weekday']

    # Deterministic columns are known ahead of time, their actual values can be used.
    deterministic_columns = [
        'Client.ID',  # subsumed by the geographic flags
        'group_prior_mean',
        'previous_reading',
        'accum_rain', #added to try to capture storm events
        'Collection_Time', # mostly missing values but may still be of some use
        '12hrPressureChange', # overnight pressure change

        #'precipIntensity',
        #'precipIntensityMax',
        #'temperatureMin',
        #'temperatureMax',
        #'humidity',
        #'windSpeed',
        #'cloudCover',

        #'flag_geographically_a_north_beach',
        'categorical_beach_grouping'
        #'12th_previous',
        #'Montrose_previous',
        #'Rainbow_previous',
        #'63rd_previous',
        #'Osterman_previous'
    ]

    # Deterministic columns are known ahead of time, their actual values are used.
    # These hourly variables have an additional parameter which defines what hours
    # should be used. For example, an entry
    #   'temperature':[-16,-13,-12,-11,-9,-3,0]
    # would indicate that the hourly temperature at offsets of
    # [-16,-13,-12,-11,-9,-3,0] from MIDNIGHT the day of should be included as
    # variables in the model.
    deterministic_hourly_columns = {
        'temperature':np.linspace(-19,4,num=6,dtype=np.int64),#range(-19,5),
        'windVectorX':np.linspace(-19,4,num=6,dtype=np.int64),#range(-19,5),#[-4,-2,0,2,4],
        'windVectorY':np.linspace(-19,4,num=6,dtype=np.int64),
        #'windSpeed':[-2,0,2,4],
        #'windBearing':[-2,0,2,4],
        'pressure':[0],
        'cloudCover':[-15], #range(-19,5),
        'humidity':[4],
        #'precipIntensity':[4]#np.linspace(-10,4,num=4,dtype=np.int64)
    }
    for var in deterministic_hourly_columns:
        for hr in deterministic_hourly_columns[var]:
            deterministic_columns.append(var + '_hour_' + str(hr))

    # Historical columns have their previous days' values added to the predictors,
    # but not the current day's value(s) unless the historical column also exists
    # in the deterministic columns list.
    # Similar to the hourly columns, you need to specify which previous days
    # to include as variables. For example, below we have an entry
    #   'temperatureMax': range(1,4)
    # which indicates that the max temperature from 1, 2, and 3 days previous
    # should be included.
    historical_columns = {
        #'temperatureMin': range(2,3),
        'temperatureMax': range(2,5),
        # 'humidity': range(1,3),
         #'windSpeed': range(1,3),
         'pressure': range(1,3),
         'dewPoint': range(1,3),
         #'cloudCover': range(1,3),
         'windVectorX': range(2,3),
         'windVectorY': range(2,3),
         'Escherichia.coli': range(2,8)
    }
    historical_columns_list = list(historical_columns.keys())


    ######################################################
    #### Get relevant columns, add historical data
    ######################################################

    all_columns = meta_columns + deterministic_columns + historical_columns_list #+ derived_columns
    all_columns = list(set(all_columns))

    df = df[all_columns]

    for var in historical_columns:
        df = rd.add_column_prior_data(
            df, var, historical_columns[var],
            beach_col_name='Client.ID', timestamp_col_name='Full_date'
        )

    df.drop((set(historical_columns_list) - set(deterministic_columns)) - set(meta_columns),
            axis=1, inplace=True)


    ######################################################
    #### Average the historical columns, fill in NaNs
    ######################################################

    # Creates a "trailing_average_daily_" column for each historical variable
    # which is simply the mean of the previous day columns of that variable.
    # NaN values for any previous day data is filled in by that mean value.
    for var in historical_columns:
        cname = 'trailing_average_daily_' + var
        rnge = historical_columns[var]
        if len(rnge) == 1:  # no need to create a trailing average of a single number...
            continue
        df[cname] = df[[str(n) + '_day_prior_' + var for n in rnge]].mean(1)
        for n in rnge:
            df[str(n) + '_day_prior_' + var].fillna(df[cname], inplace=True)

    # Do a similar process for the hourly data.
    for var in deterministic_hourly_columns:
        cname = 'trailing_average_hourly_' + var
        rnge = deterministic_hourly_columns[var]
        if len(rnge) == 1:  # no need to create a trailing average of a single number...
            continue
        df[cname] = df[[var + '_hour_' + str(n) for n in rnge]].mean(1)
        for n in rnge:
            df[var + '_hour_' + str(n)].fillna(df[cname], inplace=True)
            
    

    ######################################################
    #### Process non-numeric columns
    ######################################################

    # process all of the nonnumeric columns
    # This method just assigns a numeric value to each possible value
    # of the non-numeric column. Note that this will not work well
    # for regression-style models, where instead dummy columns should
    # be created.
    def nonnumericCols(data, verbose=True):
        for f in data.columns:
            if data[f].dtype=='object':
                if (verbose):
                    print('Column ' + str(f) + ' being treated as non-numeric')
                lbl = sklearn.preprocessing.LabelEncoder()
                lbl.fit(list(data[f].values))
                data.loc[:,f] = lbl.transform(list(data[f].values))
        return data

    # Do this at the end so meta_data has Beach names and Weekdays
    #df = nonnumericCols(df) 

    # As a last NaN filling measure, we fill the NaNs of all columns
    # that are NOT the E. coli column with the mean value of the column,
    # the mean value taken over all data not from the same year as the
    # year of the row we are filling. For example, if there is a NaN
    # in the temperatureMax column in some row from 2010, then we will
    # fill that value with the mean temperatureMax value from all years
    # that are NOT 2010.
    cols = df.columns.tolist()
    cols.remove('Escherichia.coli')
    years = df['Full_date'].map(lambda x: x.year)
    for yr in years.unique():
        not_yr = np.array(years != yr)
        is_yr = np.array(years == yr)
        df.ix[is_yr, cols] = df.ix[is_yr, cols].fillna(df.ix[not_yr, cols].median())


    ######################################################
    #### Drop any rows that still have NA, set up outputs
    ######################################################

    # The following lines will print the % of rows that:
    #  (a) have a NaN value in some column other than Escherichia.coli, AND
    #  (b) the column Escherichia.coli is NOT NaN.
    # Since we are now filling NaNs with column averages above, this should
    # always report 0%. I'm leaving the check in here just to be sure, though.
    total_rows_predictors = df.dropna(subset=['Escherichia.coli'], axis=0).shape[0]
    nonnan_rows_predictors = df.dropna(axis=0).shape[0]
    print('Dropping {0:.4f}% of rows because predictors contain NANs'.format(
        100.0 - 100.0 * nonnan_rows_predictors / total_rows_predictors
    ))

    # Any rows that still have NaNs are NaN b/c there is no E. coli reading
    # We should drop these rows b/c there is nothing for us to predict.
    df.dropna(axis=0, inplace=True)
    #df.dropna(axis=0, how='any', subset=['Full_date','Escherichia.coli'], inplace=True)

    predictors = df.drop(set(meta_columns)-set(['Client.ID']) , axis=1)
    meta_info = df[meta_columns]

    predictors = nonnumericCols(predictors)

    return predictors, meta_info



def display_predictions_by_beach(results, predict_col = 'predictedEPA'):
    '''
    Helper function to test ensemble of models on 2015 data.
    Displays the prediction results by beach, sorted from north to south.
    Parameters
    ----------
    results : dataframe with all predictions 
    Returns
    -------
    precision : percent of reported warnings that are actually correct
    recall    : percent of all actual ecoli outbreaks that are warned about
    
    Also prints table of results to console
    '''
    results['correct_warning'] = (results['expected'])&(results[predict_col])
    results['incorrect_warning'] = (results['expected']==False)&(results[predict_col])
    results['missed_warning'] = (results['expected'])&(~results[predict_col])
    print(results.groupby(['Client.ID','BEACH'])['incorrect_warning','correct_warning','missed_warning'].sum())
    TP = results['correct_warning'].sum()
    FP = results['incorrect_warning'].sum()  
    FN = results['missed_warning'].sum()  
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    return precision, recall
    
    
def calibrateThreshold(target, predictions, FNR):
    '''
    Helper function to calibrate the decision threshold such that 
    False Negative Rate (FNR) should be values between 1.0 and 10
    '''
    countOfAllNeg = len(target[target<236])
    cut = max(np.exp(predictions))
    for firstcut in np.linspace(cut,50,10):
        countOfCorrectNeg = len(target[(target<236)&(np.exp(predictions)<firstcut)])
        specif = countOfCorrectNeg/countOfAllNeg
        if specif < (1.0-FNR/100): 
            cut = firstcut + (max(np.exp(predictions0))-50)/9 #go back up one cut to begin next search
            break
    for secondcut in np.linspace(cut, cut/2,100):
        countOfCorrectNeg = len(target[(target<236)&(np.exp(predictions)<secondcut)])
        specif = countOfCorrectNeg/countOfAllNeg
        if specif <= (1.0-FNR/100): 
            cut = secondcut 
            break
    return cut






###############################################################################
###############################################################################
    ####   This builds the set of leave one out RF and GBM models   ####
###############################################################################
###############################################################################

if __name__ == '__main__':
    '''
    This script will produce and put in a folder named model_<model_suffix>:
        -- 18 .pkl files
            - 9 Random Forest Classifier models
            - 9 Gradient Boosting Regression models          
        -- 5 .csv files
            - data saved from reading in via read_data()
            - processed data ready for modeling
            - accompaning meta data for modeling
            - summary of precision, recall and tuned threshold values
            - results of testing on 2015 data
        -- 1 .txt file containing a list of predictors
    '''
    # Command Line Argument parsing
    parser = argparse.ArgumentParser(description='Process beach data.')
    parser.add_argument('-id', '--input_data', type=str,
                        metavar='data', 
                        help='input pre-read data CSV filename')
    parser.add_argument('-ip', '--input_processed', type=str,
                        metavar='processed', 
                        help='input processed modeling data CSV filename')
    parser.add_argument('-ip2', '--input_meta', type=str,
                        metavar='processed_meta', 
                        help='input processed modeling metadata CSV filename')
    parser.add_argument('-s', '--suffix', type=str,
                        metavar='model_suffix', 
                        help='suffix to identify this model build results')                        
    parser.add_argument('-v', '--verbose', action='count', default=1)

    args = parser.parse_args()
    
    if args.suffix:
        model_suffix = args.suffix
    else:
        model_suffix = time.strftime("%d_%m_%Y")
    
    directory = 'model_'+model_suffix    
    if not os.path.exists(directory):
        os.makedirs(directory)

    ##########################    
    ###   Load the data   
    ##########################
    if args.input_data:
        print('Loading data from {0}'.format(args.input_data))
        df = pd.read_csv(args.input_data, parse_dates='Full_date', low_memory=False)
        df['Full_date'] = rd.date_lookup(df['Full_date'])
    else:
        print('Reading and loading data. Saving to {}'.format(directory+'/all_data.csv'))        
        df = rd.read_data(read_weather_station=False, read_water_sensor=False, add_each_beach_data=True)
        df.to_csv(directory+'/all_data.csv', index=False)
               
    ###############################   
    ###   Prepare Predictors  
    ###############################
    if args.input_processed:
        print('Using Preprocessed data from {0} and {1}'.format(args.input_processed, args.input_meta ))
        datafilename = args.input_processed
        metadatafilename = args.input_meta
        data_processed = pd.read_csv(datafilename)
        meta_info = pd.read_csv(metadatafilename, parse_dates='Full_date')      
        meta_info['Full_date'] =  rd.date_lookup(meta_info['Full_date'])
    else:
        print('Preparing data for modeling. Saving to {0} and {1}'.format(directory+'/processed.csv', directory+'/meta_processed.csv'))
        data_processed, meta_info = prepare_data(df)
        data_processed.to_csv(directory+'/processed.csv', index=False)
        meta_info.to_csv(directory+'/meta_processed.csv', index=False)

    f = open(directory+'/feature_list.txt', 'w')
    f.write("\n".join(list(data_processed.columns) ) ) # For easy reference
    f.close()    
    if args.verbose>=1:
        print('Using the following columns as predictors:')
        for c in data_processed.columns:
            print('\t' + str(c))

        
    ##########################################################################   
    ###   Split data into Train/Validate (2006-2014) and Testing (2015)
    ##########################################################################
    train_processed = data_processed[meta_info['Full_date'] < '1-1-2015'].copy()
    test_processed  = data_processed[meta_info['Full_date'] > '1-1-2015'].copy()
    train_meta_info = meta_info[meta_info['Full_date'] < '1-1-2015'].copy()
    test_meta_info  = meta_info[meta_info['Full_date'] > '1-1-2015'].copy()

    ########################################################################## 
    ###   Setup Random Forest classifier and Gradient Boosting Regressor
    ########################################################################## 

    RF_reg = RandomForestRegressor(n_estimators=500, 
                                    max_depth=10, 
                                    max_features=0.8,
                                    min_samples_split=10,  
                                    min_samples_leaf=4, 
                                    oob_score=True, 
                                    n_jobs=-1)

    
    gbm_reg = GradientBoostingRegressor(loss='quantile', 
                                        learning_rate=0.025, 
                                        n_estimators=1500, # train longer, no concern of overfitting
                                        subsample=0.8, 
                                        min_samples_split=10, 
                                        min_samples_leaf=4, 
                                        max_depth=10, 
                                        alpha=0.85)                                  
                                          
    ########################################################################## 
    ###   Train models by holding one year out
    ###   Validate and tune cutoff thresholds on held out year                                      
    ##########################################################################       
    dataSeries = []
    colIndexes = []
    timestamps = train_meta_info['Full_date'].map(lambda x: x.year)
    print('\nBegining training and validation of hold-one-year-out models\n')    
    for yr in range(2006, 2015):
        ### HOLD OUT YEAR         
        train_ind = np.array((timestamps != yr))
        # Remove weekends from training b/c sampled under different conditions
        train_data   = train_processed.ix[train_ind & (train_meta_info['Weekday']!='Saturday')
                                                    & (train_meta_info['Weekday']!='Sunday')]
        train_target = train_meta_info.ix[train_ind & (train_meta_info['Weekday']!='Saturday')
                                              & (train_meta_info['Weekday']!='Sunday'),'Escherichia.coli']
        # Leave weekends in held out validation data
        test_data    = train_processed.ix[~train_ind]
        test_target  = train_meta_info.ix[~train_ind,'Escherichia.coli']                                    

        ### TRAIN Random Forest Regressor model and save as pickle file
        startTime = time.time() ## This is only to keep track of training time
        RF_reg.fit(train_data, np.log(train_target+1) )
        filename = directory+'/RF_regress' + '_' +str(yr) +'.pkl'
        joblib.dump(RF_reg, filename, compress=9)
        ### VALIDATE MODEL on held out year to calibarate cutoff threshold based on False Negative Rate
        predictions0 = getattr(RF_reg, 'predict')(test_data) 
        # rescales to between 0 and 1 in order to use in precision_recall_curve()
        predictionsX0= predictions0-predictions0.min() 
        predictionsX0= predictionsX0/(predictions0.max()-predictions0.min()) 
        precisionV, recallV, threshV = precision_recall_curve(test_target>=236, predictionsX0)
        threshV = np.exp(threshV*(predictions0.max()-predictions0.min())+predictions0.min()) # map back from [0,1] to origial scaling
        RFthresh = calibrateThreshold(test_target, predictions0, 2.0)  # FNR of 2%
        threshIdx = (np.abs(threshV-RFthresh)).argmin() 
        RF_rec = recallV[threshIdx]
        RF_prec = precisionV[threshIdx]
        RFthreshAlt = calibrateThreshold(test_target, predictions0, 5.0)  # FNR of 5%
        threshIdx = (np.abs(threshV-RFthreshAlt)).argmin() 
        RF_recAlt = recallV[threshIdx]
        RF_precAlt = precisionV[threshIdx]    
        # REPORT Results    
        print('  RF ensemble {0} model: thresh for 2% FNR = {1}, recall= {2}, precision  = {3}'\
              .format(yr,np.int(RFthresh),np.int(RF_rec*100+.4),np.int(RF_prec*100+.4) ))
        print('  RF ensemble {0} model: thresh for 5% FNR = {1}, recall= {2}, precision  = {3}'\
              .format(yr,np.int(RFthreshAlt),np.int(RF_recAlt*100+.4),np.int(RF_precAlt*100+.4) ))    
        if args.verbose>=3: 
            print('\t runtime of building and testing RF model was {0} minutes'.format(np.round((time.time() - startTime)/60) )) 

        ### TRAIN Gradient Boosting Regression model and save as pickle file
        startTime = time.time()
        gbm_reg.fit(train_data, np.log(train_target+1))
        filename = directory+'/GBM_regress' + '_'  + str(yr) +'.pkl'
        joblib.dump(gbm_reg, filename, compress=9)
        ### VALIDATE MODEL on held out year to calibarate cutoff threshold based on False Negative Rate
        predictions0 = getattr(gbm_reg, 'predict')(test_data) 
        # rescales to between 0 and 1 in order to use in precision_recall_curve()
        predictionsX0= predictions0-predictions0.min() 
        predictionsX0= predictionsX0/(predictions0.max()-predictions0.min()) 
        precisionV, recallV, threshV = precision_recall_curve(test_target>=236, predictionsX0)
        threshV = np.exp(threshV*(predictions0.max()-predictions0.min())+predictions0.min()) # map back from [0,1] to origial scaling
        GBMthresh = calibrateThreshold(test_target, predictions0, 2.0) # FNR of 2%
        threshIdx = (np.abs(threshV-GBMthresh)).argmin() 
        GBM_rec = recallV[threshIdx]
        GBM_prec = precisionV[threshIdx]
        GBMthreshAlt = calibrateThreshold(test_target, predictions0, 5.0)  # FNR of 5%
        threshIdx = (np.abs(threshV-GBMthreshAlt)).argmin() 
        GBM_recAlt = recallV[threshIdx]
        GBM_precAlt = precisionV[threshIdx]
        # REPORT Results    
        print('  GBM ensemble {0} model: thresh for 2% FNR = {1}, recall= {2}, precision  = {3}'\
              .format(yr,np.int(GBMthresh),np.int(GBM_rec*100+.4),np.int(GBM_prec*100+.4) ))
        print('  GBM ensemble {0} model: thresh for 5% FNR = {1}, recall= {2}, precision  = {3}'\
              .format(yr,np.int(GBMthreshAlt),np.int(GBM_recAlt*100+.4),np.int(GBM_precAlt*100+.4) ))
        if args.verbose>=3: 
            print('\t runtime of building and testing GBM model was {0} minutes'.format(np.round((time.time() - startTime)/60)))                
              
        # SAVE the precision, recall, and tuned thresholds
        d = { 'RF_precision2p':RF_prec, 'RF_recall2p':RF_rec, 'RF_thresh2p': RFthresh,
              'RF_precision5p':RF_precAlt, 'RF_recall5p':RF_recAlt, 'RF_thresh5p': RFthreshAlt,
              'GBM_precision2p':GBM_prec, 'GBM_recall2p':GBM_rec, 'GBM_thresh2p': GBMthresh,
              'GBM_precision5p':GBM_precAlt, 'GBM_recall5p':GBM_recAlt, 'GBM_thresh5p': GBMthreshAlt
              }              
        d = pd.Series(d, index = [ 'RF_precision2p', 'RF_recall2p', 'RF_thresh2p',
                                   'RF_precision5p', 'RF_recall5p', 'RF_thresh5p',
                                  'GBM_precision2p', 'GBM_recall2p', 'GBM_thresh2p',
                                  'GBM_precision5p', 'GBM_recall5p', 'GBM_thresh5p'])
        dataSeries = dataSeries + [ d ]
        colIndexes = colIndexes + [yr]

    summaryFrame = pd.DataFrame( dataSeries , index = colIndexes)
    summaryFileName = directory+'/ValidationReport2.csv'
    summaryFrame.to_csv(summaryFileName)      


    ########################################################################## 
    ###   Test models on 2015 data                                    
    ##########################################################################     
    print('\nTesting ensemble of models on 2015 data\n')       
    results = test_meta_info.copy()
    results['expected'] = results['Escherichia.coli']>=235
    results['predictedEPA'] = results['Drek_Prediction']>=235 
    
    RF_cols = []
    GBM_cols = []
    RF_bool_cols2p = []
    RF_bool_cols5p = []
    GBM_bool_cols2p = [] 
    GBM_bool_cols5p = []  
    for yr in range(2006, 2015):
        filename = directory+'/GBM_regress' + '_'  + str(yr) +'.pkl'
        gbmmodel = joblib.load(filename)
        pred_col_name = 'GBM_' +str(yr)+ '_pred'
        GBM_cols = GBM_cols + [pred_col_name]
        results[pred_col_name] = np.exp(getattr(gbmmodel, 'predict')(test_processed))
        results[pred_col_name+'_bool_2p'] = results[pred_col_name] > summaryFrame.ix[yr,'GBM_thresh2p']
        results[pred_col_name+'_bool_5p'] = results[pred_col_name] > summaryFrame.ix[yr,'GBM_thresh5p']
        GBM_bool_cols2p = GBM_bool_cols2p + [pred_col_name+'_bool_2p'] 
        GBM_bool_cols5p = GBM_bool_cols5p + [pred_col_name+'_bool_5p']
    for yr in range(2006, 2015):    
        filename = directory+'/RF_regress' + '_' +str(yr) +'.pkl'
        RFmodel = joblib.load(filename)
        pred_col_name = 'RF_' +str(yr)+ '_pred'
        results[pred_col_name] = np.exp(getattr(RFmodel, 'predict')(test_processed))  
        RF_cols = RF_cols + [pred_col_name]
        results[pred_col_name+'_bool_2p'] = results[pred_col_name] > summaryFrame.ix[yr,'RF_thresh2p']
        results[pred_col_name+'_bool_5p'] = results[pred_col_name] > summaryFrame.ix[yr,'RF_thresh5p']
        RF_cols = RF_cols + [pred_col_name]
        RF_bool_cols2p = RF_bool_cols2p + [pred_col_name+'_bool_2p']    
        RF_bool_cols5p = RF_bool_cols5p + [pred_col_name+'_bool_5p']
    results['mean_GBM'] = results[GBM_cols].mean(1)
    results['max_GBM'] = results[GBM_cols].max(1)
    results['min_GBM'] = results[GBM_cols].min(1)
    results['mean_RF'] = results[RF_cols].mean(1)
    results['max_RF'] = results[RF_cols].max(1)
    results['min_RF'] = results[RF_cols].min(1) 
    # The above results could be interesting to drill down into to see how the 
    # different models are biased, and how much variance in the predictions.
    
    # For now, the method of final prediction is to predict Ecoli_High == True 
    # IF ((any GBM predicts true) AND (any RF predicts true)) OR (EPA predicts true) 
    results['predict_RF2p'] = results[RF_bool_cols2p].sum(1) > 1
    results['predict_GBM2p'] = results[GBM_bool_cols2p].sum(1) > 1
    results['predict_Combo2p'] = (((results['predict_RF2p'])&(results['predict_GBM2p']))|(results['predictedEPA']) )
    results['predict_RF5p'] = results[RF_bool_cols5p].sum(1) > 1
    results['predict_GBM5p'] = results[GBM_bool_cols5p].sum(1) > 1
    results['predict_Combo5p'] = (((results['predict_RF5p'])&(results['predict_GBM5p']))|(results['predictedEPA']) )
    
    results.to_csv(directory+'/results_RF_GBM.csv', index=False)      
    
    # Look at performance of GMB ensemble at 5% FNR alone 
    prec, rec = display_predictions_by_beach(results, 'predict_GBM5p')
    print('GBM ensemble model at 5% FNR: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100))) 

    # Look at performance of RF ensemble at 5% FNR alone  
    prec, rec = display_predictions_by_beach(results, 'predict_RF5p')
    print('RF ensemble model at 5% FNR: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100))) 
        
    prec, rec = display_predictions_by_beach(results, 'predict_Combo5p')
    print('Combo ensemble model variant at 5% FNR with AND: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100)))        

    # Try out some variants of putting models together 
    results['predict_Combo5p'] = (((results['predict_RF5p'])|(results['predict_GBM5p']))|(results['predictedEPA']) )
    prec, rec = display_predictions_by_beach(results, 'predict_Combo5p')
    print('Combo ensemble model variant at 5% FNR with OR: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100))) 

    prec, rec = display_predictions_by_beach(results, 'predict_Combo2p')
    print('Combo ensemble model variant at 2% FNR with AND: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100)))        

    # Try out some variants of putting models together 
    results['predict_Combo2p'] = (((results['predict_RF2p'])|(results['predict_GBM2p']))|(results['predictedEPA']) )
    prec, rec = display_predictions_by_beach(results, 'predict_Combo2p')
    print('Combo ensemble model variant at 2% FNR with OR: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100)))

    # Try out some variants of putting models together 
    results['predict_RF'] = results['mean_RF']> np.exp(summaryFrame.RF_thresh5p.min()) 
    results['predict_GBM'] = results['mean_GBM']> np.exp(summaryFrame.GBM_thresh5p.min())
    results['predict_Combo'] = (((results['predict_RF'])&(results['predict_GBM']))|(results['predictedEPA']) )
    prec, rec = display_predictions_by_beach(results, 'predict_Combo')
    print('Combo ensemble model variant with one threshold: recall= {0}, precision  = {1}\n'.format(np.int(rec*100),np.int(prec*100))) 