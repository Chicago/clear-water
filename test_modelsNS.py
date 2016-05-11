
import numpy as np
import pandas as pd
import alt_read_data as rd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn import metrics
import modeling as md
import modelValidateSaver as mvs

predictors = pd.read_csv('predictors1000.csv')
meta_info = pd.read_csv('meta_info1000.csv')
meta_info['Full_date'] = rd.date_lookup(meta_info['Full_date'])
timestamps = meta_info['Full_date'].map(lambda x: x.year)


north_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()
north_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()

north_predictors.drop(['flag_geographically_a_north_beach','Rainbow_previous'], 1, inplace=True)
south_predictors.drop(['flag_geographically_a_north_beach','Montrose_previous'], 1, inplace=True)




targetsR = meta_info['Escherichia.coli']
targetsC = meta_info['Escherichia.coli']>=235
targetsL = np.log(meta_info['Escherichia.coli']+1)
targetsCn = north_meta_info['Escherichia.coli']>=235
targetsLn = np.log(north_meta_info['Escherichia.coli']+1)
targetsCs = south_meta_info['Escherichia.coli']>=235
targetsLs = np.log(south_meta_info['Escherichia.coli']+1)


modelname = 'GBMquantileAll' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)
timestamps = meta_info['Full_date'].map(lambda x: x.year)
train_ind = np.array((timestamps != yr))
results = meta_info[~train_ind].copy()

cutP = np.log(375)

predictionsGBM = getattr(model, 'predict')(predictors.ix[~train_ind,:])
results['expected'] = targetsR[~train_ind]>=235
results['predicted'] = predictionsGBM>=cutP


 #235 #.35#235 #np.log(120)#.70041157

results['BEACH'] = results['Client.ID']
results['correct_warning'] = (results['expected'])&(results['predicted'])
results['incorrect_warning'] = (results['expected']==False)&(results['predicted'])
results['missed_warning'] = (results['expected'])&(~results['predicted'])

results.groupby('BEACH')['incorrect_warning','correct_warning','missed_warning','expected'].sum()


TP = results['correct_warning'].sum()
FP = results['incorrect_warning'].sum()  
FN = results['missed_warning'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))

resultsCombined = results



modelname = 'RFClassifAll' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)


cutP = 0.55

predictionsRF = getattr(model, 'predict_proba')(predictors.ix[~train_ind,:])[:,1]
resultsCombined['expected2'] = targetsR[~train_ind]>=235
resultsCombined['predicted2'] = predictionsRF>=cutP


 #235 #.35#235 #np.log(120)#.70041157

resultsCombined['correct_warning2'] = (resultsCombined['expected2'])&(resultsCombined['predicted2'])
resultsCombined['incorrect_warning2'] = (resultsCombined['expected2']==False)&(resultsCombined['predicted2'])
resultsCombined['missed_warning2'] = (resultsCombined['expected2'])&(~resultsCombined['predicted2'])

resultsCombined.groupby('BEACH')['incorrect_warning2','correct_warning2','missed_warning2','expected2'].sum()


resultsCombined['GBM_1_RF_1'] = (results['expected'])&(results['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_1'] = (results['expected'])&(~results['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_1_RF_0'] = (results['expected'])&(results['predicted'])&(~resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_0'] = (results['expected'])&(~results['predicted'])&(~resultsCombined['predicted2'])


resultsCombined.groupby('BEACH')['GBM_1_RF_1','GBM_0_RF_1','GBM_1_RF_0','GBM_0_RF_0','expected'].sum()


resultsCombined['correct_warning2'] = (results['expected'])&((resultsCombined['predicted2'])|(results['predicted']))
resultsCombined['incorrect_warning2'] = (results['expected']==False)&((resultsCombined['predicted2'])|(results['predicted']))
resultsCombined['missed_warning2'] = (results['expected'])&((~resultsCombined['predicted2'])&(~results['predicted']))


TP = resultsCombined['correct_warning2'].sum()
FP = resultsCombined['incorrect_warning2'].sum()  
FN = resultsCombined['missed_warning2'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))






###########################################################################



timestamps_n = north_meta_info['Full_date'].map(lambda x: x.year)
train_ind_n = np.array((timestamps_n != yr))    
timestamps_s = south_meta_info['Full_date'].map(lambda x: x.year)
train_ind_s = np.array((timestamps_s != yr)) 
north_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()
north_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()

north_predictors.drop(['flag_geographically_a_north_beach','Rainbow_previous'], 1, inplace=True)
south_predictors.drop(['flag_geographically_a_north_beach','Montrose_previous'], 1, inplace=True)






modelname = 'GBMquantileSouth' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)
timestamps_n = south_meta_info['Full_date'].map(lambda x: x.year)
train_ind_n = np.array((timestamps_s != yr))
resultsS = south_meta_info[~train_ind_s].copy()

cutP = np.log(377)

predictionsGBMs = getattr(model, 'predict')(south_predictors.ix[~train_ind_s,:])
resultsS['expected'] = targetsCs
resultsS['predicted'] = predictionsGBMs>=cutP

resultsS['BEACH'] = resultsS['Client.ID']
resultsS['correct_warning'] = (resultsS['expected'])&(resultsS['predicted'])
resultsS['incorrect_warning'] = (resultsS['expected']==False)&(resultsS['predicted'])
resultsS['missed_warning'] = (resultsS['expected'])&(~resultsS['predicted'])

resultsS.groupby('BEACH')['incorrect_warning','correct_warning','missed_warning','expected'].sum()


TP = resultsS['correct_warning'].sum()
FP = resultsS['incorrect_warning'].sum()  
FN = resultsS['missed_warning'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))



resultsCombined = resultsS


modelname = 'RFClassifSouth' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)


cutP = 0.58

predictionsRFs = getattr(model, 'predict_proba')(south_predictors.ix[~train_ind_s,:])[:,1]
resultsCombined['expected2'] = targetsCs
resultsCombined['predicted2'] = predictionsRFs>=cutP


resultsCombined['correct_warning2'] = (resultsCombined['expected2'])&(resultsCombined['predicted2'])
resultsCombined['incorrect_warning2'] = (resultsCombined['expected2']==False)&(resultsCombined['predicted2'])
resultsCombined['missed_warning2'] = (resultsCombined['expected2'])&(~resultsCombined['predicted2'])

resultsCombined.groupby('BEACH')['incorrect_warning2','correct_warning2','missed_warning2','expected2'].sum()


resultsCombined['GBM_1_RF_1'] = (resultsS['expected'])&(resultsS['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_1'] = (resultsS['expected'])&(~resultsS['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_1_RF_0'] = (resultsS['expected'])&(resultsS['predicted'])&(~resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_0'] = (resultsS['expected'])&(~resultsS['predicted'])&(~resultsCombined['predicted2'])


resultsCombined.groupby('BEACH')['GBM_1_RF_1','GBM_0_RF_1','GBM_1_RF_0','GBM_0_RF_0','expected'].sum()


resultsCombined['correct_warning2'] = (resultsS['expected'])&((resultsCombined['predicted2'])|(resultsS['predicted']))
resultsCombined['incorrect_warning2'] = (resultsS['expected']==False)&((resultsCombined['predicted2'])|(resultsS['predicted']))
resultsCombined['missed_warning2'] = (resultsS['expected'])&((~resultsCombined['predicted2'])&(~resultsS['predicted']))


TP = resultsCombined['correct_warning2'].sum()
FP = resultsCombined['incorrect_warning2'].sum()  
FN = resultsCombined['missed_warning2'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))


































modelname = 'GBMquantileNorth' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)
timestamps_n = north_meta_info['Full_date'].map(lambda x: x.year)
train_ind_n = np.array((timestamps_n != yr))
resultsN = north_meta_info[~train_ind_n].copy()

cutP = np.log(258)

predictionsGBMn = getattr(model, 'predict')(north_predictors.ix[~train_ind_n,:])
resultsN['expected'] = targetsCn
resultsN['predicted'] = predictionsGBMn>=cutP

resultsN['BEACH'] = resultsN['Client.ID']
resultsN['correct_warning'] = (resultsN['expected'])&(resultsN['predicted'])
resultsN['incorrect_warning'] = (resultsN['expected']==False)&(resultsN['predicted'])
resultsN['missed_warning'] = (resultsN['expected'])&(~resultsN['predicted'])

resultsN.groupby('BEACH')['incorrect_warning','correct_warning','missed_warning','expected'].sum()


TP = resultsN['correct_warning'].sum()
FP = resultsN['incorrect_warning'].sum()  
FN = resultsN['missed_warning'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))



resultsCombined = resultsN


modelname = 'RFClassifNorth' # 'RFreglog' # 'RFreg'# 'GBMhuber2' #'RFClassif'# 'GBMquantile2' #'GBMhuber2'  
yr = 2009
filename = modelname + '_' + str(yr) +'.pkl'
model = joblib.load(filename)


cutP = 0.39

predictionsRFn = getattr(model, 'predict_proba')(north_predictors.ix[~train_ind_n,:])[:,1]
resultsCombined['expected2'] = targetsCn
resultsCombined['predicted2'] = predictionsRFn>=cutP


resultsCombined['correct_warning2'] = (resultsCombined['expected2'])&(resultsCombined['predicted2'])
resultsCombined['incorrect_warning2'] = (resultsCombined['expected2']==False)&(resultsCombined['predicted2'])
resultsCombined['missed_warning2'] = (resultsCombined['expected2'])&(~resultsCombined['predicted2'])

resultsCombined.groupby('BEACH')['incorrect_warning2','correct_warning2','missed_warning2','expected2'].sum()


resultsCombined['GBM_1_RF_1'] = (resultsN['expected'])&(resultsN['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_1'] = (resultsN['expected'])&(~resultsN['predicted'])&(resultsCombined['predicted2'])
resultsCombined['GBM_1_RF_0'] = (resultsN['expected'])&(resultsN['predicted'])&(~resultsCombined['predicted2'])
resultsCombined['GBM_0_RF_0'] = (resultsN['expected'])&(~resultsN['predicted'])&(~resultsCombined['predicted2'])


resultsCombined.groupby('BEACH')['GBM_1_RF_1','GBM_0_RF_1','GBM_1_RF_0','GBM_0_RF_0','expected'].sum()


resultsCombined['correct_warning2'] = (resultsN['expected'])&((resultsCombined['predicted2'])|(resultsN['predicted']))
resultsCombined['incorrect_warning2'] = (resultsN['expected']==False)&((resultsCombined['predicted2'])|(resultsN['predicted']))
resultsCombined['missed_warning2'] = (resultsN['expected'])&((~resultsCombined['predicted2'])&(~resultsN['predicted']))


TP = resultsCombined['correct_warning2'].sum()
FP = resultsCombined['incorrect_warning2'].sum()  
FN = resultsCombined['missed_warning2'].sum()  
precision = TP/(TP+FP)
recall = TP/(TP+FN)

print('year={0}: , model: {1}, precision  = {2}'.format(yr,modelname, precision))
print('year={0}: , model: {1},  recall     = {2}'.format(yr,modelname, recall))

## try to see if mixing models gives better predictions








