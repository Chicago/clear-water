# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:23:46 2016

@author: Rebecca
"""


from numpy import *
import pandas as pd
from read_data import *


data = read_data_simplified()

dfn = data.copy()

dfn = clean_up_beaches(dfn)


dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 1)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 2)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 3)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 4)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 5)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 6)
dfn = add_column_prior_data(dfn, 'Ecoli_geomean' , 7)

col_list = ['prior_1_Ecoli_geomean','prior_2_day_Ecoli_geomean','prior_3_day_Ecoli_geomean',
            'prior_4_day_Ecoli_geomean','prior_5_day_Ecoli_geomean',
            'prior_6_day_Ecoli_geomean','prior_7_day_Ecoli_geomean']
temp = dfn[col_list].copy()
dfn['7_day_moving_avg'] = temp.mean(axis=1)


col_list = ['Timestamp','Beach','Month','Weekday','Ecoli_geomean','prior_1_day_Ecoli_geomean','7_day_moving_avg']
df = dfn[col_list].copy()
df['PriorReading'] =  pd.Series(map((  lambda x,y: y if isnan(x) else x  ), df['prior_1_day_Ecoli_geomean'], df['7_day_moving_avg'])  )  
df.drop(['prior_1_day_Ecoli_geomean'], axis=1, inplace=True)
len(df.ix[isnan(df['PriorReading'])])
df = df.dropna(axis=0, subset=['PriorReading'])
df = df.reset_index()
df = df.drop('index', axis=1)


beaches = list(pd.Series(df['Beach'].value_counts().index))
for j in range(len(beaches)):
    beach2 = df.ix[df['Beach']==beaches[j],['Timestamp','Beach','PriorReading']].reset_index()
    c = beach2.columns.tolist()
    c[c.index('PriorReading')] = beaches[j] + '_prior'
    beach2.columns = c
    beach2.drop(['index','Beach'], axis=1, inplace=True)
    df = pd.merge(df, beach2, on='Timestamp', how='left')
    
    
tmp = df[['Montrose_prior', 'Calumet_prior', '63rd_prior',
       'Rainbow_prior', 'Ohio_prior', '31st_prior', '57th_prior',
       'South Shore_prior', '12th_prior', 'Ostermann_prior', 'Loyola_prior',
       'Oak_prior', 'Foster_prior', 'North Ave_prior', 'Albion_prior',
       'Jarvis_prior', 'Rogers_prior', 'Howard_prior', 'Juneway_prior',
       '39th_prior', 'Pratt_prior']]

df['city_mean'] = tmp.mean(axis=1)
fill_value = pd.DataFrame({col: tmp.mean(axis=1) for col in tmp.columns})
df.fillna(fill_value, inplace=True)


# process all of the nonnumeric columns
def nonnumericCols(data, verbose=True):
	from sklearn import preprocessing
	for f in data.columns:
		if data[f].dtype=='object':
			if (verbose):
				print(f)
			lbl = preprocessing.LabelEncoder()
			lbl.fit(list(data[f].values))
			data[f] = lbl.transform(list(data[f].values))
	return data	
df = nonnumericCols(df)




outputs = df['Ecoli_geomean']>=235
inputs = df.drop(['Timestamp','Ecoli_geomean'], axis=1)
from sklearn.cross_validation import train_test_split
in_train, in_test, out_train, out_test = train_test_split(inputs, outputs, test_size=0.3, random_state=42)





from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier( n_estimators=1000, learning_rate=0.05, max_depth=6)
clf.fit(in_train, out_train)

from sklearn import metrics
predictions = clf.predict(in_test)
print(metrics.classification_report(out_test, predictions ))
print( metrics.confusion_matrix(out_test, predictions) )

metrics.roc_auc_score(out_test, predictions)

preds = clf.predict_proba(in_test)
fpr, tpr, thresholds = metrics.roc_curve(out_test, preds[:,1])
metrics.auc(fpr, tpr)





from sklearn.linear_model import LogisticRegression
clf_LR = LogisticRegression( )
clf_LR.fit(in_train, out_train)

predictionsLR = clf_LR.predict(in_test)
print(metrics.classification_report(out_test, predictionsLR ))
print( metrics.confusion_matrix(out_test, predictionsLR) )

predsLR = clf_LR.predict_proba(in_test)
fpr_LR, tpr_LR, thresholds = metrics.roc_curve(out_test, predsLR[:,1])
metrics.auc(fpr_LR, tpr_LR)


import matplotlib.pyplot as plt
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='GBM')
plt.plot(fpr_LR, tpr_LR, label='LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()




imp = pd.DataFrame({'features': inputs.columns, 'importance': clf.feature_importances_ } )
imp.sort_values('importance')






from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresh = metrics.precision_recall_curve(out_test, preds[:,1])
average_precision = average_precision_score(out_test, preds[:,1])


# Plot Precision-Recall curve
plt.figure(1)
plt.plot(recall, precision, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall example: AUC={0:0.2f}'.format(average_precision))
plt.legend(loc="lower left")
plt.show()





