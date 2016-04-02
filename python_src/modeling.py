import numpy as np
import sklearn
import sklearn.ensemble
import read_data as rd
import visualizations as viz
import matplotlib.pyplot as plt


def model(timestamps, predictors, classes, classifier=None, hyperparams=None, verbose=False):
    '''
    Creates several models using leave-one-year-out cross validation.

    ROC and PR curves are plotted as a side-effect.

    Parameters
    ----------
    timestamps : Nx1 pandas series of timestamps.
                 Each element should have a "year" attribute.
    predictors : NxM pandas DataFrame, all values should be numeric,
                 and there should be no NaN values.
    classes    : Nx1 array like of binary outcomes, e.g. True or False.
    classifier : sklearn classifier, should have the attributes "fit"
                 and "predict_proba" at the least.
    hyperparams: Dictionary of hyper parameters to pass to the
                 classifier method.
    verbose    : True if the clf.feature_importances_ should be printed

    Returns
    -------
    clfs : Dictionary of (year, classifier) pairs, where the classifier
           is the model found by leaving the specified year out of the
           training set.
    '''
    if classifier is None:
        classifier = sklearn.ensemble.GradientBoostingClassifier
    if hyperparams is None:
        hyperparams = {}

    timestamps = timestamps.map(lambda x: x.year)

    start = timestamps.min()
    stop = timestamps.max()

    stop = min(stop, 2014) # do not include 2015

    roc_ax = plt.subplots(1)[1]
    pr_ax = plt.subplots(1)[1]

    clfs = dict()

    for yr in range(start, stop+1):
        is_not_yr = (timestamps < yr) | (timestamps > yr)
        is_not_2007 = (timestamps != 2007)
        train_indices = np.array(is_not_yr & is_not_2007)

        clf = sklearn.base.clone(classifier(**hyperparams))
        clf.fit(predictors.ix[train_indices,:], classes[train_indices])

        clfs[yr] = clf

        predictions = clf.predict_proba(predictors.ix[~train_indices,:])[:,1]

        auc_roc = viz.roc(predictions, classes[~train_indices],
                          block_show=False, ax=roc_ax)[3]
        auc_pr = viz.precision_recall(predictions, classes[~train_indices],
                                      block_show=False, ax=pr_ax)[3]

        auc_roc = float(auc_roc)
        auc_pr = float(auc_pr)
        roc_ax.get_lines()[-2].set_label(str(yr) + ' - AUC: {0:.4f}'.format(auc_roc))
        pr_ax.get_lines()[-2].set_label(str(yr) + ' - AUC: {0:.4f}'.format(auc_pr))

        if verbose:
            print('Year ' + str(yr))
            print('Feature importances:')
            feat_imps = clf.feature_importances_
            idxs = np.argsort(feat_imps)[::-1]
            max_width = max([len(c) for c in predictors.columns])

            for c, fi in zip(predictors.columns[idxs], feat_imps[idxs]):
                print('  {0:<{1}} : {2:.5f}'.format(c, max_width+1, fi))

    return clfs, roc_ax, pr_ax


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
    meta_info  : Nx2 DataFrame containing the columns 'Escherichia.coli' and
                 'Full_date', to be used, e.g., for leave-one-year-out cross
                 validation and creating the true class labels (elevated vs.
                 not elevated E. coli levels).
    '''
    if df is None:
        df = rd.read_data()

    # Leaving 2015 as the final validation set
    df = df[df['Full_date'] < '1-1-2015']


    ######################################################
    #### Add derived columns here
    ######################################################

    # df['DayOfYear'] = df['Full_date'].map(lambda x: x.dayofyear)


    ######################################################
    #### List all columns you will use
    ######################################################

    # Meta columns are not used as predictors
    meta_columns = ['Client.ID', 'Full_date', 'Escherichia.coli']

    # Deterministic columns are known ahead of time, their actual values are used
    # with no previous days being used.
    deterministic_columns = [
        # 'Client.ID',
        'sunriseTime',
        'precipIntensity',
        'precipIntensityMax',
        'temperatureMin',
        'temperatureMax',
        'humidity',
        'windSpeed',
        'cloudCover',
        'flag_geographically_a_north_beach',
        # 'flag_geographic_group_1',
        'flag_geographic_group_2',
        # 'flag_geographic_group_3',
        # 'flag_geographic_group_4',
        'flag_geographic_group_5',
        # 'flag_geographic_group_6',
    ]
    deterministic_hourly_columns = {
        'temperature':[-16,-13,-12,-11,-9,-3,0],
        'windSpeed':[1,2,3,4],
        'windBearing':[1,2,3,4],
        'pressure':[0],
        'cloudCover':[0,2,4]
    }
    for var in deterministic_hourly_columns:
        for hr in deterministic_hourly_columns[var]:
            deterministic_columns.append(var + '_hour_' + str(hr))

    # Historical columns have their previous days' values added to the predictors,
    # but not the current day's value(s) unless the historical column also exists
    # in the deterministic columns list.
    historical_columns = {
        'temperatureMin': range(1,3),
        'temperatureMax': range(1,8),
        'humidity': range(1,3),
        'windSpeed': range(1,8),
        'cloudCover': range(1,8),
        'Escherichia.coli': range(1,8)
    }
    historical_columns_list = list(historical_columns.keys())


    ######################################################
    #### Get relevant columns, add historical data
    ######################################################

    all_columns = list(set(meta_columns + deterministic_columns + historical_columns_list))

    df = df[all_columns]

    for var in historical_columns:
        df = rd.add_column_prior_data(
            df, var, historical_columns[var],
            beach_col_name='Client.ID', timestamp_col_name='Full_date'
        )

    df.drop((set(historical_columns_list) - set(deterministic_columns)) - set(meta_columns),
            axis=1, inplace=True)


    ######################################################
    #### Average the historical column, fill in NaNs
    ######################################################

    for var in historical_columns:
        cname = 'historical_average_' + var
        rnge = historical_columns[var]
        df[cname] = df[[str(n) + '_day_prior_' + var for n in rnge]].mean(1)
        for n in rnge:
            df[str(n) + '_day_prior_' + var].fillna(df[cname], inplace=True)


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
                data[f] = lbl.transform(list(data[f].values))
        return data

    df = nonnumericCols(df)

    cols = df.columns.tolist()
    cols.remove('Escherichia.coli')
    df[cols] = df[cols].fillna(df[cols].mean())


    ######################################################
    #### Drop any rows that still have NA, set up outputs
    ######################################################

    total_rows_predictors = df.dropna(subset=['Escherichia.coli'], axis=0).shape[0]
    nonnan_rows_predictors = df.dropna(axis=0).shape[0]
    print('Dropping {0:.4f}% of rows because predictors contain NANs'.format(
        100.0 - 100.0 * nonnan_rows_predictors / total_rows_predictors
    ))

    df.dropna(axis=0, inplace=True)

    predictors = df.drop(meta_columns, axis=1)
    meta_info = df[meta_columns]

    return predictors, meta_info


if __name__ == '__main__':
    df = rd.read_data(read_weather_station=False, read_water_sensor=False)
    epa_model_df = df[['Drek_Prediction', 'Escherichia.coli']].dropna()
    predictors, meta_info = prepare_data(df)
    timestamps = meta_info['Full_date']
    classes = meta_info['Escherichia.coli'] > 235

    print('Using the following columns as predictors:')
    for c in predictors.columns:
        print('\t' + str(c))
    hyperparams = {
        # Parameters that effect computation
        'n_estimators':1000, 'max_depth':5,
        # Misc parameters
        'n_jobs':-1, 'verbose':False
    }
    clfs, roc_ax, pr_ax = model(timestamps, predictors, classes,
                                classifier=sklearn.ensemble.RandomForestClassifier,
                                hyperparams=hyperparams,
                                verbose=True)

    # Add the EPA model to the ROC and PR curves, prettify
    c = roc_ax.get_lines()
    for line in c:
        line.set_alpha(.75)

    fpr, tpr, threshes, auc_roc = viz.roc(epa_model_df['Drek_Prediction'],
                                          epa_model_df['Escherichia.coli'] > 235,
                                          ax=roc_ax, block_show=False)
    auc_roc = float(auc_roc)
    epa_line = roc_ax.get_lines()[-2]
    epa_line.set_color([0,0,0])
    epa_line.set_ls('--')
    epa_line.set_linewidth(3)
    epa_line.set_alpha(.85)
    epa_line.set_label('EPA Model - AUC: {0:.4f}'.format(auc_roc))

    i = np.where(threshes < 235.0)[0][0]

    print(fpr[i])
    print(tpr[i])
    print(threshes[i])

    roc_ax.plot(fpr[i], tpr[i], 'xk', label='EPA model at 235',
                markersize=10.0, markeredgewidth=2.0)

    roc_ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    # roc_ax.set_xlim([0, .1])
    # roc_ax.set_aspect('auto')
    roc_ax.grid(True, which='major')

    c = pr_ax.get_children()
    for line in c:
        line.set_alpha(.75)

    tpr, ppv, threshes, auc_pr = viz.precision_recall(epa_model_df['Drek_Prediction'],
                                                      epa_model_df['Escherichia.coli'] > 235,
                                                      ax=pr_ax, block_show=False)
    auc_pr = float(auc_pr)
    epa_line = pr_ax.get_lines()[-2]
    epa_line.set_color([0,0,0])
    epa_line.set_ls('--')
    epa_line.set_linewidth(3)
    epa_line.set_alpha(.85)
    epa_line.set_label('EPA Model - AUC: {0:.4f}'.format(auc_pr))

    i = np.where(threshes < 235.0)[0][0]

    print(tpr[i])
    print(ppv[i])
    print(threshes[i])

    pr_ax.plot(tpr[i], ppv[i], 'xk', label='EPA model at 235',
               markersize=10.0, markeredgewidth=2.0)


    pr_ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    pr_ax.grid(True, which='major')

    plt.draw()
    plt.show(block=True)
