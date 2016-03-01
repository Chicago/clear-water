import numpy as np
import sklearn
import sklearn.ensemble
import read_data as rd
import visualizations as viz
import pandas as pd


def main():
    # Load data, don't load water/weather station data
    df = rd.read_data(read_water_sensor=False, read_weather_station=False)

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
    df.drop('city_mean', axis=1, inplace=True)

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

    train_index = np.array(df['Full_date'] < '1-1-2015')
    Y = df['Escherichia.coli'] > 235
    df.drop(['Escherichia.coli', 'Full_date'], axis=1, inplace=True)

    clf = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6, verbose=True)
    clf.fit(df.ix[train_index,:], Y[train_index])

    predictions = clf.predict_proba(df.ix[~np.array(train_index),:])[:,1]

    viz.roc(predictions, Y[~train_index], block_show=False)
    viz.precision_recall(predictions, Y[~train_index], block_show=True)

    return clf, df, Y, train_index


if __name__ == '__main__':
    main()
