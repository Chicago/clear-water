import requests
import pandas as pd
import json
import re
import read_data as rd
import math
import numpy as np
import daily_weather as dw
import datetime
from sklearn.externals import joblib


resp = requests.get('https://data.cityofchicago.org/api/views/2ivx-z93u/rows.json?accessType=DOWNLOAD')
theJSON = json.loads(resp.text)
data = []
for item in theJSON["data"]:
	data.append(item[9:14])

df=pd.DataFrame(data)


for d in range(len(df)):
	ts=datetime.datetime.strptime(df.iloc[d,0],'%Y-%m-%dT%H:%M:%S')
	df.loc[d,'Collection_Time'] = int(ts.strftime('%H'))*60+int(ts.strftime('%M'))

	
	
df[0]=df[0].str.extract('(....-..-..)',expand=False)
df.columns=['Full_date','Beach','Reading1','Reading2','Mean','Collection_Time']
df= df.sort_values(by=['Beach','Full_date'],ascending=[True,False])
df= df.loc[(df.Beach != "Columbia")&(df.Beach!="Lane")&(df.Beach!="Loyola")&(df.Beach!="Marion Mahoney Griffin")&(df.Beach!="North Shore")&(df.Beach!="NA")]
temp = df
df=df.groupby('Beach').first()

df.reset_index(level=0,inplace=True)

days_prior_ecoli = 8
temp = temp.groupby('Beach').head(days_prior_ecoli).reset_index(drop=True)
for number in range(days_prior_ecoli):
	df['%d_day_prior_Escherichia.coli'%(number)]=np.nan

i=0

for beaches in range(math.ceil(len(temp)/days_prior_ecoli)) :
	for days in range(days_prior_ecoli):
		df.loc[beaches,'%d_day_prior_Escherichia.coli'%(days)]=float(temp.loc[i,'Mean'])
		i+=1
df =df.drop('0_day_prior_Escherichia.coli',axis=1)


df['Client.ID']=np.nan
df['lat']=np.nan
df['long']=np.nan
df['Group']=np.nan
df['North']=np.nan
latlong = pd.read_csv('Beaches_LatLong.csv',dtype={'Client.ID':object,'Latitude':str,'Longitude':str,'Group':str,'North':str})
for i in range(len(df['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df.loc[i,'Beach']= latlong.loc[j,'Beach']
			df.loc[i,'lat'] = latlong.loc[j,'Latitude']
			df.loc[i,'long'] = latlong.loc[j,'Longitude']
			df.loc[i,'Client.ID']= latlong.loc[j,'Client.ID']
			df.loc[i,'Group'] = latlong.loc[j,'Group']
			df.loc[i,'North'] = latlong.loc[j,'North']
			
apikey= 'abc7928ccbb6a65eeba40bc7c10e1529'

newcols_today = ('humidity_hour_4','pressure_hour_0','temperature_hour_0','temperature_hour_4','windVectorX_hour_0','windVectorX_hour_4','windVectorY_hour_0','windVectorY_hour_4','pressure_hour_4','temperature_hour_1','temperature_hour_2','temperature_hour_3','windSpeed_hour_4','precipIntensity_hour_0','windBearing_hour_4')

for var in newcols_today :
	df[var]=np.nan
	
newcols_yest = ('1_day_prior_dewPoint','1_day_prior_pressure','cloudCover_hour_-15','temperature_hour_-19','temperature_hour_-19','temperature_hour_-18','temperature_hour_-17','temperature_hour_-16','temperature_hour_-15','temperature_hour_-14','temperature_hour_-13','temperature_hour_-12','temperature_hour_-11','temperature_hour_-10','temperature_hour_-9','temperature_hour_-8','temperature_hour_-7','temperature_hour_-6','temperature_hour_-5','temperature_hour_-4','temperature_hour_-3','temperature_hour_-2','temperature_hour_-1','windVectorX_hour_-19','windVectorX_hour_-14','windVectorX_hour_-9', 'windVectorX_hour_-5', 'windVectorY_hour_-19','windVectorY_hour_-14','windVectorY_hour_-9', 'windVectorY_hour_-5','pressure_hour_-8','Max_precipIntensity-1','1_day_prior_temperatureMax','windSpeed','humidity','1_day_prior_temperatureMin', 'cloudCover')

for var in newcols_yest :
	df[var]=np.nan

newcols_twodays = ('2_day_prior_dewPoint', '2_day_prior_pressure', '2_day_prior_temperatureMax','2_day_prior_windVectorX','2_day_prior_windVectorY','Max_precipIntensity-2','2_day_prior_temperatureMin')

for var in newcols_twodays :
	df[var]=np.nan
	
df['3_day_prior_temperatureMax']= np.nan
df['4_day_prior_temperatureMax']= np.nan

	
for i in range(len(df)) :
	lattitude = str(df.loc[i,'lat'])
	longitude = str(df.loc[i,'long'])
	for day in range(5) :
		d = pd.to_datetime(df.loc[i,'Full_date'],format = '%Y-%m-%d')-datetime.timedelta(days=day)
		d = datetime.datetime.strptime(str(d),'%Y-%m-%d %H:%M:%S')
		weather = dw.daily_weather(lat= lattitude, long=longitude,date=d.strftime("%Y-%m-%dT%H:%M:%S"),apikey=apikey)
		if day == 0 :
			for item in newcols_today :
				if item =='humidity_hour_4' :
					df.loc[i,item]= weather.loc[0,'4.humidity.hourly']
				elif item == 'pressure_hour_0' :
					df.loc[i,item]= weather.loc[0,'0.pressure.hourly']
				elif item == 'temperature_hour_0' :
					df.loc[i,item]= weather.loc[0,'0.temperature.hourly']
				elif item == 'temperature_hour_1' :
					df.loc[i,item]= weather.loc[0,'1.temperature.hourly']
				elif item == 'temperature_hour_2' :
					df.loc[i,item]= weather.loc[0,'2.temperature.hourly']
				elif item == 'temperature_hour_3' :
					df.loc[i,item]= weather.loc[0,'3.temperature.hourly']
				elif item == 'temperature_hour_4' :
					df.loc[i,item]= weather.loc[0,'4.temperature.hourly']
				elif item == 'windVectorX_hour_0' :
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'0.windBearing.hourly']/360)*weather.loc[0,'0.windSpeed.hourly']
				elif item == 'windVectorX_hour_4':
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'4.windBearing.hourly']/360)*weather.loc[0,'4.windSpeed.hourly']
				elif item == 'windVectorY_hour_0':
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'0.windBearing.hourly']/360)*weather.loc[0,'0.windSpeed.hourly']
				elif item == 'windVectorY_hour_4' :
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'4.windBearing.hourly']/360)*weather.loc[0,'4.windSpeed.hourly']
				elif item == 'pressure_hour_4' :
					df.loc[i,item] = weather.loc[0,'4.pressure.hourly']
				elif item == 'windSpeed_hour_4' :
					df.loc[i,item] = weather.loc[0,'4.windSpeed.hourly']
				elif item == 'cloudCover_hour_4' :
					df.loc[i,item] = weather.loc[0,'4.cloudCover.hourly']
				elif item == 'windSpeed_hour_1' :
					df.loc[i,item] = weather.loc[0,'1.windSpeed.hourly']
				elif item == 'precipIntensity_hour_0':
					df.loc[i,item] = weather.loc[0,'0.precipIntensity.hourly']
				elif item == 'windSpeed_hour_3' :
					df.loc[i,item] = weather.loc[0,'3.windSpeed.hourly']
				elif item == 'windBearing_hour_4' :
					df.loc[i,item] = weather.loc[0,'4.windBearing.hourly']
		elif day == 1 :
			for item in newcols_yest :
				if item == '1_day_prior_dewPoint':
					df.loc[i,item] = weather.loc[0,'dewPoint']
				elif item == '1_day_prior_presure':
					df.loc[i,item] = weather.loc[0,'pressure']
				elif item == 'cloudCover_hour_-15':
					df.loc[i,item] = weather.loc[0,'9.cloudCover.hourly']
				elif item == 'temperature_hour_-19':
					df.loc[i,item] = weather.loc[0,'5.temperature.hourly']
				elif item == 'temperature_hour_-18':
					df.loc[i,item] = weather.loc[0,'6.temperature.hourly']
				elif item == 'temperature_hour_-17':
					df.loc[i,item] = weather.loc[0,'7.temperature.hourly']
				elif item == 'temperature_hour_-16':
					df.loc[i,item] = weather.loc[0,'8.temperature.hourly']
				elif item == 'temperature_hour_-15':
					df.loc[i,item] = weather.loc[0,'9.temperature.hourly']
				elif item == 'temperature_hour_-14':
					df.loc[i,item] = weather.loc[0,'10.temperature.hourly']
				elif item == 'temperature_hour_-13':
					df.loc[i,item] = weather.loc[0,'11.temperature.hourly']
				elif item == 'temperature_hour_-12':
					df.loc[i,item] = weather.loc[0,'12.temperature.hourly']
				elif item == 'temperature_hour_-11':
					df.loc[i,item] = weather.loc[0,'13.temperature.hourly']
				elif item == 'temperature_hour_-10':
					df.loc[i,item] = weather.loc[0,'14.temperature.hourly']
				elif item == 'temperature_hour_-9':
					df.loc[i,item] = weather.loc[0,'15.temperature.hourly']
				elif item == 'temperature_hour_-8':
					df.loc[i,item] = weather.loc[0,'16.temperature.hourly']
				elif item == 'temperature_hour_-7':
					df.loc[i,item] = weather.loc[0,'17.temperature.hourly']
				elif item == 'temperature_hour_-6':
					df.loc[i,item] = weather.loc[0,'18.temperature.hourly']
				elif item == 'temperature_hour_-5':
					df.loc[i,item] = weather.loc[0,'19.temperature.hourly']
				elif item == 'temperature_hour_-4':
					df.loc[i,item] = weather.loc[0,'20.temperature.hourly']
				elif item == 'temperature_hour_-3':
					df.loc[i,item] = weather.loc[0,'21.temperature.hourly']
				elif item == 'temperature_hour_-2':
					df.loc[i,item] = weather.loc[0,'22.temperature.hourly']
				elif item == 'temperature_hour_-1':
					df.loc[i,item] = weather.loc[0,'23.temperature.hourly']
				elif item == 'windVectorX_hour_-19':
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'5.windBearing.hourly']/360)*weather.loc[0,'5.windSpeed.hourly']
				elif item == 'windVectorX_hour_-14':
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'10.windBearing.hourly']/360)*weather.loc[0,'10.windSpeed.hourly']
				elif item == 'windVectorX_hour_-9':
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'15.windBearing.hourly']/360)*weather.loc[0,'15.windSpeed.hourly']
				elif item == 'windVectorX_hour_-5':
					df.loc[i,item]=np.cos(np.pi*2*weather.loc[0,'19.windBearing.hourly']/360)*weather.loc[0,'19.windSpeed.hourly']
				elif item == 'windVectorY_hour_-19':
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'5.windBearing.hourly']/360)*weather.loc[0,'5.windSpeed.hourly']
				elif item == 'windVectorY_hour_-14':
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'10.windBearing.hourly']/360)*weather.loc[0,'10.windSpeed.hourly']
				elif item == 'windVectorY_hour_-9':
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'15.windBearing.hourly']/360)*weather.loc[0,'15.windSpeed.hourly']
				elif item == 'windVectorY_hour_-5':
					df.loc[i,item]=np.sin(np.pi*2*weather.loc[0,'19.windBearing.hourly']/360)*weather.loc[0,'19.windSpeed.hourly']
				elif item == 'pressure_hour_-8' :
					df.loc[i,item] = weather.loc[0,'16.pressure.hourly']
				elif item == 'Max_precipIntensity-1':
					df.loc[i,item] = weather.loc[0,'precipIntensityMax']
				elif item == '1_day_prior_temperatureMax':
					df.loc[i,item] = weather.loc[0,'temperatureMax']
				elif item == '1_day_prior_pressure':
					df.loc[i,item] = weather.loc[0,'pressure']
				elif item == 'windSpeed' :
					df.loc[i,item] = weather.loc[0,'windSpeed']
				elif item == 'humidity' :
					df.loc[i,item] = weather.loc[0,'humidity']
				elif item == '1_day_prior_temperatureMin' :
					df.loc[i,item] = weather.loc[0,'temperatureMin']
				elif item == 'cloudCover' :
					df.loc[i,item] = weather.loc[0,'cloudCover']
		elif day == 2:
			for item in newcols_twodays :
				if item == '2_day_prior_dewPoint':
					df.loc[i,item] = weather.loc[0,'dewPoint']
				elif item == '2_day_prior_pressure':
					df.loc[i,item] = weather.loc[0,'pressure']
				elif item == '2_day_prior_temperatureMax':
					df.loc[i,item] = weather.loc[0,'temperatureMax']
				elif item == '2_day_prior_windVectorX' :
					df.loc[i,'2_day_prior_windVectorX'] = np.cos(np.pi*2*weather.loc[0,'windBearing']/360)*weather.loc[0,'windSpeed']
				elif item == '2_day_prior_windVectorY' :
					df.loc[i,'2_day_prior_windVectorY'] = np.sin(np.pi*2*weather.loc[0,'windBearing']/360)*weather.loc[0,'windSpeed']
				elif item == 'Max_precipIntensity-2':
					df.loc[i,item] = weather.loc[0,'precipIntensityMax']
				elif item == '2_day_prior_temperatureMin':
					df.loc[i,item] = weather.loc[0,'temperatureMin']
		elif day == 3 :
			df.loc[i,'3_day_prior_temperatureMax'] = weather.loc[0,'temperatureMax']
			df.loc[i,'Max_precipIntensity-3'] = weather.loc[0,'precipIntensityMax']
			df.loc[i,'3_day_prior_temperatureMin'] = weather.loc[0,'temperatureMin']
		elif day == 4 :
			df.loc[i,'4_day_prior_temperatureMax'] = weather.loc[0,'temperatureMax'] 
			df.loc[i,'Max_precipIntensity-4'] = weather.loc[0,'precipIntensityMax']
			df.loc[i,'4_day_prior_temperatureMin'] = weather.loc[0,'temperatureMin'] 

df['12hrPressureChange'] = df['pressure_hour_4']-df['pressure_hour_-8']			

group_means = pd.DataFrame()
group_means['Group']=range(1,7)

for i in range(int(df['Group'].min())-1,int(df['Group'].max())) :
	group_means.loc[i,'group_prior_mean']=df.groupby('Group')['1_day_prior_Escherichia.coli'].mean()[i]
	
for row in range(len(df['Group'])) :
	df.loc[row,'group_prior_mean'] = group_means.loc[int(df.loc[row,'Group'])-1,'group_prior_mean'] 

df['accum_rain'] = df[['Max_precipIntensity-1','Max_precipIntensity-2','Max_precipIntensity-3','Max_precipIntensity-4']].sum(axis=1)

df['trailing_average_daily_temperatureMax'] = df[['1_day_prior_temperatureMax','2_day_prior_temperatureMax','3_day_prior_temperatureMax','4_day_prior_temperatureMax']].mean(axis=1)

df['trailing_average_daily_temperatureMin'] = df[['1_day_prior_temperatureMin','2_day_prior_temperatureMin','3_day_prior_temperatureMin','4_day_prior_temperatureMin']].mean(axis=1)

df['trailing_average_daily_Escherichia.coli']= df[['1_day_prior_Escherichia.coli','2_day_prior_Escherichia.coli',
'3_day_prior_Escherichia.coli',
'4_day_prior_Escherichia.coli',
'5_day_prior_Escherichia.coli',
'6_day_prior_Escherichia.coli',
'7_day_prior_Escherichia.coli']].mean(axis=1)

df['trailing_average_daily_pressure']= df[['1_day_prior_pressure','2_day_prior_pressure']].mean(axis=1)

df['trailing_average_daily_dewPoint']= df[['1_day_prior_dewPoint','2_day_prior_dewPoint']].mean(axis=1)

df['trailing_average_hourly_windVectorX'] = df[['windVectorX_hour_4','windVectorX_hour_0','windVectorX_hour_-5','windVectorX_hour_-9','windVectorX_hour_-14','windVectorX_hour_-19']].mean(axis=1)

df['trailing_average_hourly_windVectorY'] = df[['windVectorY_hour_4','windVectorY_hour_0','windVectorY_hour_-5','windVectorY_hour_-9','windVectorY_hour_-14','windVectorY_hour_-19']].mean(axis=1)

df['trailing_average_hourly_temperature'] = df[['temperature_hour_4', 'temperature_hour_0', 'temperature_hour_-5','temperature_hour_-9', 'temperature_hour_-14', 'temperature_hour_-19']].mean(axis=1)

model_cols = ('Client.ID','windVectorX_hour_-5','windVectorY_hour_-9','group_prior_mean','windVectorY_hour_0','temperature_hour_4','temperature_hour_-5','temperature_hour_0', 'windVectorY_hour_4', 'accum_rain', 'categorical_beach_grouping', '12hrPressureChange', 'windVectorX_hour_0', 'temperature_hour_-19', 'windVectorX_hour_4', 'temperature_hour_-14','windVectorX_hour_-14', 'previous_reading','cloudCover_hour_-15', 'humidity_hour_4', 'windVectorX_hour_-9','windVectorY_hour_-19','windVectorY_hour_-5', 'Collection_Time', 'windVectorX_hour_-19', 'pressure_hour_0', 'temperature_hour_-9', 'windVectorY_hour_-14','2_day_prior_Escherichia.coli', '3_day_prior_Escherichia.coli', '4_day_prior_Escherichia.coli', '5_day_prior_Escherichia.coli', '6_day_prior_Escherichia.coli', '7_day_prior_Escherichia.coli', '2_day_prior_temperatureMax', '3_day_prior_temperatureMax', '4_day_prior_temperatureMax','2_day_prior_windVectorX', '2_day_prior_windVectorY','1_day_prior_pressure', '2_day_prior_pressure', '1_day_prior_dewPoint', '2_day_prior_dewPoint','trailing_average_daily_Escherichia.coli','trailing_average_daily_temperatureMax','trailing_average_daily_pressure', 'trailing_average_daily_dewPoint', 'trailing_average_hourly_temperature','trailing_average_hourly_windVectorX','trailing_average_hourly_windVectorY')

# model_cols2 = ('temperature_hour_-3','precipIntensity','windSpeed','humidity_hour_4','temperature_hour_-10','temperature_hour_2','windSpeed_hour','temperature_hour_0','temperature_hour_-15','temperature_hour_-14','cloudCover_hour_4','temperatureMax','temperature_hour_-11','temperature_hour_-8','temperature_hour_-13','temperature_hour_-7','humidity','categorical_beach_grouping','temperature_hour_-12','temperature_hour_-4','temperature_hour_-19','windSpeed_hour_1','temperature_hour_-17','temperature_hour_3','temperatureMin','temperature_hour_-9','temperature_hour_-5','temperature_hour_-1','precipIntensity_hour_0', 'cloudCover','pressure_hour_0', 'windSpeed_hour_2','precipIntensity_hour_4','windSpeed_hour_3','flag_geographically_a_north_beach','temperature_hour_-2','temperature_hour_-16','temperature_hour_-6','precipIntensityMax','DayOfYear','temperature_hour_4','windBearing_hour_4','temperature_hour_1','temperature_hour_-18','1_day_prior_temperatureMax','2_day_prior_temperatureMax','3_day_prior_temperatureMax','1_day_prior_temperatureMin','2_day_prior_temperatureMin','1_day_prior_Escherichia.coli','2_day_prior_Escherichia.coli','3_day_prior_Escherichia.coli', '4_day_prior_Escherichia.coli','5_day_prior_Escherichia.coli','6_day_prior_Escherichia.coli','7_day_prior_Escherichia.coli', 'trailing_average_daily_temperatureMax','trailing_average_daily_temperatureMin', 'trailing_average_daily_Escherichia.coli','trailing_average_hourly_windSpeed','trailing_average_hourly_precipIntensity', 'trailing_average_hourly_temperature', '1_day_prior_Escherichia.coli_beach_in_grouping_1', '1_day_prior_Escherichia.coli_beach_in_grouping_2', '1_day_prior_Escherichia.coli_beach_in_grouping_3', '1_day_prior_Escherichia.coli_beach_in_grouping_4', '1_day_prior_Escherichia.coli_beach_in_grouping_5', '1_day_prior_Escherichia.coli_beach_in_grouping_6')

df=df.rename(columns={'Group':'categorical_beach_grouping','1_day_prior_Escherichia.coli':'previous_reading'})


model_df = pd.DataFrame()
for cols in model_cols :
	model_df[cols] = df[cols]

model_df = model_df[pd.notnull(model_df['Client.ID'])]

rf_preds = pd.DataFrame()
rf_preds['Beach'] = df.Beach[pd.notnull(df['Client.ID'])]

gbm_preds = pd.DataFrame()
gbm_preds['Beach'] = df.Beach[pd.notnull(df['Client.ID'])]
for yr in range (2006,2015) :
	filename = ('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/model_26_08_2016/RF_regress_'+str(yr)+'.pkl')

	rfmodel = joblib.load(filename)

	rf_preds['rf_'+str(yr)]=np.exp(getattr(rfmodel, 'predict')(model_df))
	
	filename = ('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/model_26_08_2016/gbm_regress_'+str(yr)+'.pkl')

	gbmmodel = joblib.load(filename)

	gbm_preds['gbm_'+str(yr)]=np.exp(getattr(gbmmodel, 'predict')(model_df))

report = pd.read_csv('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/model_26_08_2016/ValidationReport2.csv',dtype={'RF_thresh2p':np.float64,'RF_thresh5p':np.float64,'GBM_thresh2p':np.float64,'GBM_thresh5p':np.float64})
predict = pd.DataFrame()
predict['Beach']=df.Beach[pd.notnull(df['Client.ID'])]
predict['RF_Predictions']=rf_preds.mean(axis=1)
predict['RF_thresh2p']=report.RF_thresh2p.mean(axis=0)
predict['RF_thresh5p']=report.RF_thresh5p.mean(axis=0)
predict['GBM_Predictions']=gbm_preds.mean(axis=1)
predict['GBM_thresh2p']=report.GBM_thresh2p.mean(axis=0)
predict['GBM_thresh5p']=report.GBM_thresh5p.mean(axis=0)

