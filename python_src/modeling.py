import numpy as np
import sklearn
import sklearn.ensemble
import read_data as rd
import visualizations as viz
import matplotlib.pyplot as plt


def model(timestamps, predictors, classes, classifier=None, hyperparams=None):
    '''
    Creates several GBMs using leave-one-year-out cross validation.

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
        train_indices = np.array((timestamps < yr) | (timestamps > yr))

        clf = sklearn.ensemble.GradientBoostingClassifier(**hyperparams)
        clf.fit(predictors.ix[train_indices,:], classes[train_indices])

        clfs[yr] = clf

        predictions = clf.predict_proba(predictors.ix[~train_indices,:])[:,1]

        viz.roc(predictions, classes[~train_indices], block_show=False, ax=roc_ax)
        viz.precision_recall(predictions, classes[~train_indices], block_show=False, ax=pr_ax)

    return clfs


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

    df['DayOfYear'] = df['Full_date'].map(lambda x: x.dayofyear)


    ######################################################
    #### List all columns you will use
    ######################################################

    # Meta columns are not used as predictors
    meta_columns = ['Full_date', 'Escherichia.coli']

    # Deterministic columns are known ahead of time, their actual values are used
    # with no previous days being used.
    deterministic_columns = [
        'Client.ID', 'Weekday', 'sunriseTime', 'DayOfYear',
        'windSpeed_hour_4', 'windBearing_hour_4'
    ]

    # Historical columns have their previous days' values added to the predictors,
    # but not the current day's value(s). The value NUM_LOOKBACK_DAYS set below
    # controls the number of previous days added. Nothing is currently done to
    # fill NA values here, so if you wish to use columns with a high rate of data
    # loss, then you should add logic to fill the NA values.
    historical_columns = [
        # 'precipIntensity', 'precipIntensityMax',
        # 'temperatureMin', 'temperatureMax',
        # 'humidity', 'windSpeed', 'cloudCover'
    ]

    # Each historical column will have the data from 1 day back, 2 days back,
    # ..., NUM_LOOKBACK_DAYS days back added to the predictors.
    NUM_LOOKBACK_DAYS = 3


    ######################################################
    #### Get relevant columns, add historical data
    ######################################################

    all_columns = meta_columns + deterministic_columns + historical_columns

    df = df[all_columns]

    df = rd.add_column_prior_data(
        df, historical_columns, range(1, NUM_LOOKBACK_DAYS + 1),
        beach_col_name='Client.ID', timestamp_col_name='Full_date'
    )

    df.drop(historical_columns, axis=1, inplace=True)


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


    ######################################################
    #### Drop any rows that still have NA, set up outputs
    ######################################################

    df.dropna(axis=0, inplace=True)

    predictors = df.drop(['Escherichia.coli', 'Full_date'], axis=1)
    meta_info = df[['Escherichia.coli', 'Full_date']]

    return predictors, meta_info


if __name__ == '__main__':
    predictors, meta_info = prepare_data()
    timestamps = meta_info['Full_date']
    classes = meta_info['Escherichia.coli'] > 235

    print('Using the following columns as predictors:')
    for c in predictors.columns:
        print('\t' + str(c))
    hyperparams = {
        # 'n_estimators':100, 'learning_rate':0.05,
        # 'max_depth':4, 'subsample':0.9, 'verbose':True
    }
    clfs = model(timestamps, predictors, classes,
                 classifier=sklearn.ensemble.RandomForestClassifier,
                 hyperparams=hyperparams)

    df2 = rd.read_data()
    df2 = df2[['Drek_Prediction', 'Escherichia.coli']].dropna()

    # TODO: better document/automate this plotting business.
    N = 18

    ax = plt.figure(1).get_axes()[0]
    viz.roc(df2['Drek_Prediction'], df2['Escherichia.coli'] > 235,
            ax=ax, block_show=False)
    c = ax.get_children()
    for i in range(N):
        c[i].set_alpha(.5)
    c[N].set_color([0,0,0])
    c[N].set_ls('--')
    c[N].set_linewidth(3)
    c[N].set_alpha(.8)
    ax.legend([c[i] for i in range(0,N+2,2)],
              ['06', '07', '08', '09'] + [str(i) for i in range(10,15)] + ['EPA Model'],
              loc=4)


    ax = plt.figure(2).get_axes()[0]
    viz.precision_recall(df2['Drek_Prediction'], df2['Escherichia.coli'] > 235,
                         ax=ax, block_show=False)
    c = ax.get_children()
    for i in range(N):
        c[i].set_alpha(.5)
    c[N].set_color([0,0,0])
    c[N].set_ls('--')
    c[N].set_linewidth(3)
    c[N].set_alpha(.8)
    ax.legend([c[i] for i in range(0,N+2,2)],
              ['06', '07', '08', '09'] + [str(i) for i in range(10,15)] + ['EPA Model'],
              loc=1)

    plt.draw()
    plt.show(block=True)
