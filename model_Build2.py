
import numpy as np
import pandas as pd
import alt_read_data as rd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn import metrics
import modeling as md
import modelValidateSaver as mvs


complete_data = pd.read_csv('completeCleanData.csv', parse_dates='Full_date', low_memory=False)
complete_data['Full_date'] = rd.date_lookup(complete_data['Full_date'])


data = complete_data[complete_data['Full_date'] < '1-1-2015'].copy()
data = data.ix[(data['Weekday']!='Saturday')&(data['Weekday']!='Sunday')]
data = data.reset_index()
data.drop(['index'], 1, inplace=True)

predictors, meta_info = md.prepare_data(data)

predictors.to_csv('predictors1000.csv', index=False)
meta_info.to_csv('meta_info1000.csv',index=False)



north_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_predictors = predictors.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()
north_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==1)].copy()
south_meta_info = meta_info.ix[(predictors['flag_geographically_a_north_beach']==0)].copy()

north_predictors.drop(['flag_geographically_a_north_beach','Rainbow_previous'], 1, inplace=True)
south_predictors.drop(['flag_geographically_a_north_beach','Montrose_previous'], 1, inplace=True)






timestamps = data['Full_date'].map(lambda x: x.year)
start = timestamps.min()
stop = timestamps.max()
stop = min(stop, 2014) 

targetsR = meta_info['Escherichia.coli']
targetsC = meta_info['Escherichia.coli']>=235
targetsL = np.log(meta_info['Escherichia.coli']+1)
targetsCn = north_meta_info['Escherichia.coli']>=235
targetsLn = np.log(north_meta_info['Escherichia.coli']+1)
targetsCs = south_meta_info['Escherichia.coli']>=235
targetsLs = np.log(south_meta_info['Escherichia.coli']+1)



RF_clf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=10,  min_samples_leaf=6,
                                oob_score=True, n_jobs=-1, class_weight={0: 1.0, 1: 1/.15})

gbm_reg_q = GradientBoostingRegressor(loss='quantile', learning_rate=0.05, n_estimators=1000, subsample=0.8, 
                                      min_samples_split=6, min_samples_leaf=6, max_depth=10, alpha=0.90)           
                                      




dataSeries = []
colIndexes = []




for yr in range(start, stop+1):
    timestamps = meta_info['Full_date'].map(lambda x: x.year)
    train_ind = np.array((timestamps != yr))
    timestamps_n = north_meta_info['Full_date'].map(lambda x: x.year)
    train_ind_n = np.array((timestamps_n != yr))    
    timestamps_s = south_meta_info['Full_date'].map(lambda x: x.year)
    train_ind_s = np.array((timestamps_s != yr)) 

    RF_prec, RF_rec, RFthresh = mvs.model_validate_Classifier(train_ind, predictors, targetsC , yr, RF_clf, 'RFclassifAll')
    GBM_prec_q, GBM_rec_q, GBMthresh = mvs.model_validate_RegressionLog(train_ind, predictors, targetsL , yr, gbm_reg_q, 'GBMquantileAll')    
    RF_precN, RF_recN, RFthreshN = mvs.model_validate_Classifier(train_ind_n, north_predictors, targetsCn , yr, RF_clf, 'RFclassifNorth')
    GBM_prec_qN, GBM_rec_qN, GBMthreshN = mvs.model_validate_RegressionLog(train_ind_n, north_predictors, targetsLn , yr, gbm_reg_q, 'GBMquantileNorth')
    RF_precS, RF_recS, RFthreshS = mvs.model_validate_Classifier(train_ind_s, south_predictors, targetsCs , yr, RF_clf, 'RFclassifSouth')
    GBM_prec_qS, GBM_rec_qS, GBMthreshS = mvs.model_validate_RegressionLog(train_ind_s, south_predictors, targetsLs , yr, gbm_reg_q, 'GBMquantileSouth')

    
    d = { 'RF_prec':RF_prec, 'RF_rec':RF_rec, 'RFthresh':RFthresh,
         'GBM_prec_q':GBM_prec_q, 'GBM_rec_q':GBM_rec_q, 'GBMthresh':GBMthresh,
         'RF_precN':RF_precN, 'RF_recN':RF_recN, 'RFthreshN':RFthreshN,
         'GBM_prec_qN':GBM_prec_qN, 'GBM_rec_qN':GBM_rec_qN, 'GBMthreshN':GBMthreshN,
         'RF_precS':RF_precS, 'RF_recS':RF_recS, 'RFthreshS':RFthreshS,
         'GBM_prec_qS':GBM_prec_qS, 'GBM_rec_qS':GBM_rec_qS, 'GBMthreshS':GBMthreshS
          }
    d = pd.Series(d, index = [ 'RF_prec', 'RF_rec', 'RFthresh',
                              'RF_precN', 'RF_recN', 'RFthreshN',
                              'RF_precS', 'RF_recS', 'RFthreshS',
                              'GBM_prec_q', 'GBM_rec_q', 'GBMthresh',
                              'GBM_prec_qN', 'GBM_rec_qN', 'GBMthreshN',
                              'GBM_prec_qS', 'GBM_rec_qS', 'GBMthreshS' ])
    dataSeries = dataSeries + [ d ]
    colIndexes = colIndexes + [yr]

summaryFrame = pd.DataFrame( dataSeries , index = colIndexes) 


summaryFrame.to_csv('RF_GBM_results11.csv')



#####################################################################

from sklearn.metrics import precision_recall_curve
modelnames = ['RFclassifAll','RFclassifNorth','RFclassifSouth', 'GBMquantileAll','GBMquantileNorth','GBMquantileSouth']



for yr in range(start, stop+1):

    #yr = 2006

    timestamps = meta_info['Full_date'].map(lambda x: x.year)
    train_ind = np.array((timestamps != yr))
    timestamps_n = north_meta_info['Full_date'].map(lambda x: x.year)
    train_ind_n = np.array((timestamps_n != yr))    
    timestamps_s = south_meta_info['Full_date'].map(lambda x: x.year)
    train_ind_s = np.array((timestamps_s != yr)) 


    filename = 'RFclassifAll_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict_proba')(predictors.ix[~train_ind,:])[:,1]
    precisionV, recallV, thresh = precision_recall_curve(targetsC[~train_ind], predictions)
    RF_prec = precisionV[recallV>.15].max()
    RF_rec = recallV[precisionV==RF_prec][0]
    RFthresh = thresh[precisionV[:-1]==RF_prec][0] 
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, RF_prec))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, RF_rec))
    
    filename = 'RFclassifNorth_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict_proba')(north_predictors.ix[~train_ind_n,:])[:,1]
    precisionV, recallV, thresh = precision_recall_curve(targetsCn[~train_ind_n], predictions)
    RF_precN = precisionV[recallV>.15].max()
    RF_recN = recallV[precisionV==RF_precN][0]
    RFthreshN = thresh[precisionV[:-1]==RF_precN][0]     
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, RF_precN))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, RF_recN))

    filename = 'RFclassifSouth_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict_proba')(south_predictors.ix[~train_ind_s,:])[:,1]
    precisionV, recallV, thresh = precision_recall_curve(targetsCs[~train_ind_s], predictions)
    RF_precS = precisionV[recallV>.15].max()
    RF_recS = recallV[precisionV==RF_precS][0]
    RFthreshS = thresh[precisionV[:-1]==RF_precS][0]    
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, RF_precS))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, RF_recS)) 
    
    filename = 'GBMquantileAll_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict')(predictors.ix[~train_ind,:])
    predictionsX= predictions-predictions.min()
    predictionsX= predictionsX/predictions.max()
    precisionV, recallV, thresh = precision_recall_curve(targetsL[~train_ind]>np.log(235), predictionsX)
    GBM_prec_q = precisionV[recallV>.15].max()
    GBM_rec_q = recallV[precisionV==GBM_prec_q][0]
    GBMthresh = thresh[precisionV[:-1]==GBM_prec_q][0]  
    GBMthresh = GBMthresh*predictions.max()+predictions.min()    
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, GBM_prec_q))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, GBM_rec_q))   
   
    filename = 'GBMquantileNorth_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict')(north_predictors.ix[~train_ind_n,:])
    predictionsX= predictions-predictions.min()
    predictionsX= predictionsX/predictions.max()
    precisionV, recallV, thresh = precision_recall_curve(targetsLn[~train_ind_n]>np.log(235), predictionsX)
    GBM_prec_qN = precisionV[recallV>.15].max()
    GBM_rec_qN = recallV[precisionV==GBM_prec_qN][0]
    GBMthreshN = thresh[precisionV[:-1]==GBM_prec_qN][0]
    GBMthreshN = GBMthreshN*predictions.max()+predictions.min()    
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, GBM_prec_qN))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, GBM_rec_qN)) 
    
    filename = 'GBMquantileSouth_' + str(yr) +'.pkl'
    model = joblib.load(filename)
    predictions = getattr(model, 'predict')(south_predictors.ix[~train_ind_s,:])
    predictionsX= predictions-predictions.min()
    predictionsX= predictionsX/predictions.max()
    precisionV, recallV, thresh = precision_recall_curve(targetsLs[~train_ind_s]>np.log(235), predictionsX)
    GBM_prec_qS = precisionV[recallV>.15].max()
    GBM_rec_qS = recallV[precisionV==GBM_prec_qS][0]
    GBMthreshS = thresh[precisionV[:-1]==GBM_prec_qS][0]
    GBMthreshS = GBMthreshS*predictions.max()+predictions.min()    
    print('year={0}: , model: {1}, precision  = {2}'.format(yr,filename, GBM_prec_qS))
    #print('year={0}: , model: {1},  recall     = {2}'.format(yr,filename, GBM_rec_qS))     
    
    d = { 'RF_prec':RF_prec, 'RF_rec':RF_rec, 'RFthresh':RFthresh,
         'GBM_prec_q':GBM_prec_q, 'GBM_rec_q':GBM_rec_q, 'GBMthresh':GBMthresh,
         'RF_precN':RF_precN, 'RF_recN':RF_recN, 'RFthreshN':RFthreshN,
         'GBM_prec_qN':GBM_prec_qN, 'GBM_rec_qN':GBM_rec_qN, 'GBMthreshN':GBMthreshN,
         'RF_precS':RF_precS, 'RF_recS':RF_recS, 'RFthreshS':RFthreshS,
         'GBM_prec_qS':GBM_prec_qS, 'GBM_rec_qS':GBM_rec_qS, 'GBMthreshS':GBMthreshS
          }
    d = pd.Series(d, index = ['RF_prec', 'RF_rec', 'RFthresh',
                              'RF_precN', 'RF_recN', 'RFthreshN',
                              'RF_precS', 'RF_recS', 'RFthreshS',
                              'GBM_prec_q', 'GBM_rec_q', 'GBMthresh',
                              'GBM_prec_qN', 'GBM_rec_qN', 'GBMthreshN',
                              'GBM_prec_qS', 'GBM_rec_qS', 'GBMthreshS' ])
    dataSeries = dataSeries + [ d ]
    colIndexes = colIndexes + [yr]

summaryFrame = pd.DataFrame( dataSeries , index = colIndexes) 


summaryFrame.to_csv('RF_GBM_results12.csv')








