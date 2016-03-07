import numpy as np
import sklearn
import sklearn.ensemble
import read_data as rd
import visualizations as viz
import pandas as pd
import matplotlib.pyplot as plt


def gbm(timestamps, predictors, classes):

    timestamps = timestamps.map(lambda x: x.year)

    start = timestamps.min()
    stop = timestamps.max()
    stop = min(stop, 2014) # do not include 2015

    roc_ax = plt.subplots(1)[1]
    pr_ax = plt.subplots(1)[1]

    clfs = dict()

    for yr in range(start, stop+1):
        train_indices = np.array((timestamps < yr) | (timestamps > yr))

        clf = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=500, learning_rate=0.05,
            max_depth=6, subsample=0.7, verbose=True
        )
        clf.fit(predictors.ix[train_indices,:], classes[train_indices])

        clfs[yr] = clf

        predictions = clf.predict_proba(predictors.ix[~train_indices,:])[:,1]

        viz.roc(predictions, classes[~train_indices], block_show=False, ax=roc_ax)
        viz.precision_recall(predictions, classes[~train_indices], block_show=False, ax=pr_ax)

    return clfs


def prepare_data(df=None):
    if df is None:
        df = rd.read_data()

    df = df[df['Full_date'] < '1-1-2015']

    df = df[['Full_date','Client.ID','Weekday','Escherichia.coli']]

    df2 = rd.add_column_prior_data(df, 'Escherichia.coli', range(1,8),
                                   beach_col_name='Client.ID', timestamp_col_name='Full_date')

    df2['7_day_moving_avg'] = df2[df2.columns[df2.columns.map(lambda x: x[1:6]) == '_day_']].mean(axis=1)

    df = pd.merge(df, df2[['Full_date', 'Client.ID', '7_day_moving_avg']],
                  how='left', on=['Full_date', 'Client.ID'])

    df['PriorEColiReading'] =  pd.Series(map((lambda x,y: y if np.isnan(x) else x),
                                             df2['1_day_prior_Escherichia.coli'], df['7_day_moving_avg']))

    for b in df['Client.ID'].unique().tolist():
        beach2 = df.ix[df['Client.ID']==b,['Full_date','Client.ID','PriorEColiReading']].reset_index()
        c = beach2.columns.tolist()
        c[c.index('PriorEColiReading')] = b + '_PriorEColiReading'
        beach2.columns = c
        beach2.drop(['index','Client.ID'], axis=1, inplace=True)
        df = pd.merge(df, beach2, on='Full_date', how='left')

    priorEColiColumns = [b + '_PriorEColiReading' for b in df['Client.ID'].unique().tolist()]
    df['city_mean'] = df[priorEColiColumns].mean(axis=1)
    fill_value = pd.DataFrame({col: df['city_mean'] for col in priorEColiColumns})
    df.fillna(fill_value, inplace=True)

    # process all of the nonnumeric columns
    def nonnumericCols(data, verbose=True):
    	for f in data.columns:
    		if data[f].dtype=='object':
    			if (verbose):
    				print(f)
    			lbl = sklearn.preprocessing.LabelEncoder()
    			lbl.fit(list(data[f].values))
    			data[f] = lbl.transform(list(data[f].values))
    	return data
    df = nonnumericCols(df)
    df.dropna(axis=0, inplace=True)

    return df


if __name__ == '__main__':
    df = prepare_data()
    timestamps = df['Full_date']
    classes = df['Escherichia.coli'] > 235
    predictors = df.drop(['Full_date', 'Escherichia.coli'], axis=1)
    clfs = gbm(timestamps, predictors, classes)

    df2 = rd.read_data()
    # need to investigate why it's _x, what merge caused that?
    df2 = df2[['Drek_Prediction_x', 'Escherichia.coli']].dropna()

    plt.figure(1)
    ax = plt.gca()
    viz.roc(df2['Drek_Prediction_x'], df2['Escherichia.coli'] > 235,
            ax=ax, block_show=False)
    c = ax.get_children()
    for i in range(18):
        c[i].set_alpha(.5)
    c[18].set_color([0,0,0])
    c[18].set_ls('-.')
    c[18].set_linewidth(3)
    c[18].set_alpha(.8)
    ax.legend([c[i] for i in range(0,20,2)],
              ['06', '07', '08', '09'] + [str(i) for i in range(10,15)] + ['EPA Model'],
              loc=4)


    plt.figure(2)
    ax = plt.gca()
    c = ax.get_children()
    viz.precision_recall(df2['Drek_Prediction_x'], df2['Escherichia.coli'] > 235,
                         ax=ax, block_show=False)
    for i in range(18):
        c[i].set_alpha(.5)
    c[18].set_color([0,0,0])
    c[18].set_ls('-.')
    c[18].set_linewidth(3)
    c[18].set_alpha(.8)
    ax.legend([c[i] for i in range(0,20,2)],
              ['06', '07', '08', '09'] + [str(i) for i in range(10,15)] + ['EPA Model'],
              loc=1)

    plt.draw()
    plt.show(block=True)
