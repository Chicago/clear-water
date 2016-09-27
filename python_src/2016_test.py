# This program is to determine how well the models would do on a long term basis, not just a day to day basis. Basically it is an extension of daily.py without having to use DarkSky.io and pay for it. This is very similar to daily.py Differences between daily.py and this program are:
# An individual beach's weather is not loaded for each day. Instead we use but a centralized location's weather for all the analyses.

import pandas as pd
import json
import requests
import datetime
import re
import numpy as np
from datetime import date, timedelta as td
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt
import math
######################################################
####DataFrames used in this program
######################################################

#weather - This comes from a .csv that output from datelist.py
 
#df - The main Data frame that all necessary variables will be stored to

#daily_group_means - Contains mean of each group of beaches for each day in the date range.

#model_df - The data frame we run the RF and GBM models from.

#svc_df -  The data frame we run the SVC models from.

#rf_preds - Data Frame to hold the RF Predictions. This will have the predictions for each RF model from build_models.py

#gbm_preds - Data Frame to hold the GBM Predictions. This will have the predictions for each GBM model from build_models.py 

#SVC_preds - Data Frame to hold the SVC Predictions. This will have the predictions for the svc model from SVC_regression.py

#test - The concatenated dataframe of the all the GBM, SVC, and RF models with the actual outcomes of the prediction day. 

##############################################################
#### Read in a csv that has the weather from forecast.io
#### The CSV can be created from datelist.py
##############################################################
weather = pd.DataFrame()
weather = pd.read_csv("2016_weather.csv",encoding='latin1')
first_day = weather.loc[0,'Date'] #Day that we are going to start on.
last_day = weather.loc[len(weather)-1,'Date'] #Final date in that is going to be analyzed.

######################################################
#### Read in data from the city website
######################################################
resp = requests.get('https://data.cityofchicago.org/api/views/2ivx-z93u/rows.json?accessType=DOWNLOAD')
theJSON = json.loads(resp.text)
data = []
for item in theJSON["data"]:
	data.append(item[9:14]) #Grab (Timestamp, Beach, Sample 1&2, and Mean columns from city website.)

#The table df will be the main data frame to store all the variables for before doing the final models.
df=pd.DataFrame(data)

#Split apart the timestamp column
for d in range(len(df)):
	ts=datetime.datetime.strptime(df.iloc[d,0],'%Y-%m-%dT%H:%M:%S')
	df.loc[d,'Collection_Time'] = int(ts.strftime('%H'))*60+int(ts.strftime('%M')) #The 'Collection time' in final models are in minutes after midnight.

######################################################
####Populating the main DataFrame
######################################################	
	
df[0]=df[0].str.extract('(....-..-..)',expand=False) #Create 'Full_date' in first column 

df.columns=['Full_date','Beach','Reading1','Reading2','Mean','Collection_Time'] #Rename first 6 columns in df

df= df.sort_values(by=['Beach','Full_date'],ascending=[True,False]) #Sort by 'Beach' and 'Full_date' to organize data better.

df= df.loc[(df.Beach != "Columbia")&(df.Beach!="Lane")&(df.Beach!="Loyola")&(df.Beach!="Marion Mahoney Griffin")&(df.Beach!="North Shore")&(df.Beach!="NA")] #Get rid of all the data from beaches that we do not analyze. 

df = df[(df.Full_date >= first_day)&(df.Full_date <= last_day)] #Trim df down to the days we are going to be looking for at each beach.

df = df.reset_index(drop=True) #The index is of the rows where the days we are analyzing were pulled from, before we trimmed df down. So we reset the whole index to just give us the rows that we are using.

latlong = pd.read_csv('Beaches_LatLong.csv',dtype={'Client.ID':object,'Latitude':str,'Longitude':str,'Group':str,'North':str}) #Using the Beaches_LatLong.csv we read in all that data, for the 'Group' and 'North' variables.
#Add latlong to df
for i in range(len(df['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df.loc[i,'Beach']= latlong.loc[j,'Beach']
			df.loc[i,'Client.ID']= latlong.loc[j,'Client.ID']
			df.loc[i,'Group'] = latlong.loc[j,'Group']
			df.loc[i,'North'] = latlong.loc[j,'North']

weather = weather.set_index('Date') #Set the index of the weather data frame by the date to make it easier to search for rows.

# df_beach= df.loc[:,['Full_date','Beach','Mean']]
# df_beach = df_beach.set_index(['Full_date','Beach'])
# df_beach.sortlevel(inplace=True)


newcols_today = ('humidity_hour_4','pressure_hour_0','temperature_hour_0','temperature_hour_4','windVectorX_hour_0','windVectorX_hour_4','windVectorY_hour_0','windVectorY_hour_4','pressure_hour_4','temperature_hour_1','temperature_hour_2','temperature_hour_3','windSpeed_hour_4','precipIntensity_hour_0','windBearing_hour_4') #These are going to be the columns of weather for the specific day that we are predicting in the models.


newcols_yest = ('1_day_prior_dewPoint','1_day_prior_pressure','cloudCover_hour_-15','temperature_hour_-19','temperature_hour_-19','temperature_hour_-18','temperature_hour_-17','temperature_hour_-16','temperature_hour_-15','temperature_hour_-14','temperature_hour_-13','temperature_hour_-12','temperature_hour_-11','temperature_hour_-10','temperature_hour_-9','temperature_hour_-8','temperature_hour_-7','temperature_hour_-6','temperature_hour_-5','temperature_hour_-4','temperature_hour_-3','temperature_hour_-2','temperature_hour_-1','windVectorX_hour_-19','windVectorX_hour_-14','windVectorX_hour_-9', 'windVectorX_hour_-5', 'windVectorY_hour_-19','windVectorY_hour_-14','windVectorY_hour_-9', 'windVectorY_hour_-5','pressure_hour_-8','Max_precipIntensity-1','1_day_prior_temperatureMax','windSpeed','humidity','1_day_prior_temperatureMin', 'cloudCover') #These are going to be the columns of weather for the day before that we are predicting in the models.


newcols_twodays = ('2_day_prior_dewPoint', '2_day_prior_pressure', '2_day_prior_temperatureMax','2_day_prior_windVectorX','2_day_prior_windVectorY','Max_precipIntensity-2','2_day_prior_temperatureMin') #These are going to be the columns of weather for the two days before that we are predicting in the models.

#Go through each row of df and add in the information for all the columns we created above
for n in range(len(df)):
	yesterday= df.loc[n,'Full_date'] #Retrieve the date of the row.
	ts=datetime.datetime.strptime(yesterday,'%Y-%m-%d') #Put the date into a form we can manipulate.
	day_add = ts+td(days=1) #Day we are trying to predict
	day_sub = ts-td(days=1) #2 days before we are trying to predict
	two_day_sub = ts-td(days=2)#3 days before we are trying to predict
	three_day_sub = ts-td(days=3)#4 days before we are trying to predict
	today = str(day_add.strftime('%Y-%m-%d')) #String of the date predicting.
	two_days = str(day_sub.strftime('%Y-%m-%d'))#String of 2 days before we are trying to predict
	three_days = str(two_day_sub.strftime('%Y-%m-%d'))#String of 3 days before we are trying to predict
	four_days = str(three_day_sub.strftime('%Y-%m-%d'))#String of 4 days before we are trying to predict
	###Goes through each of the columns of the individual rows of df, finds the specific cell in the weather data frame, and puts it in the correct column in df. ###
	for var in newcols_today:
		if var =='humidity_hour_4' :
			df.loc[n,var]= weather.loc[today,'4.humidity.hourly']
		elif var == 'pressure_hour_0' :
			df.loc[n,var]= weather.loc[today,'0.pressure.hourly']
		elif var == 'temperature_hour_0' :
			df.loc[n,var]= weather.loc[today,'0.temperature.hourly']
		elif var == 'temperature_hour_1' :
			df.loc[n,var]= weather.loc[today,'1.temperature.hourly']
		elif var == 'temperature_hour_2' :
			df.loc[n,var]= weather.loc[today,'2.temperature.hourly']
		elif var == 'temperature_hour_3' :
			df.loc[n,var]= weather.loc[today,'3.temperature.hourly']
		elif var == 'temperature_hour_4' :
			df.loc[n,var]= weather.loc[today,'4.temperature.hourly']
		elif var == 'windVectorX_hour_0' :
			df.loc[n,var]=np.cos(np.pi*2*weather.loc[today,'0.windBearing.hourly']/360)*weather.loc[today,'0.windSpeed.hourly'] #You have the wind speed and wind bearings from darksky.io using those you get how hard the wind is blowing in the x and y directions.
		elif var == 'windVectorX_hour_4':
			df.loc[n,var]=np.cos(np.pi*2*weather.loc[today,'4.windBearing.hourly']/360)*weather.loc[today,'4.windSpeed.hourly']
		elif var == 'windVectorY_hour_0':
			df.loc[n,var]=np.sin(np.pi*2*weather.loc[today,'0.windBearing.hourly']/360)*weather.loc[today,'0.windSpeed.hourly']
		elif var == 'windVectorY_hour_4' :
			df.loc[n,var]=np.sin(np.pi*2*weather.loc[today,'4.windBearing.hourly']/360)*weather.loc[today,'4.windSpeed.hourly']
		elif var == 'pressure_hour_4' :
			df.loc[n,var] = weather.loc[today,'4.pressure.hourly']
		elif var == 'windSpeed_hour_4' :
			df.loc[n,var] = weather.loc[today,'4.windSpeed.hourly']
		elif var == 'cloudCover_hour_4' :
			df.loc[n,var] = weather.loc[today,'4.cloudCover.hourly']
		elif var == 'windSpeed_hour_1' :
			df.loc[n,var] = weather.loc[today,'1.windSpeed.hourly']
		elif var == 'precipIntensity_hour_0':
			df.loc[n,var] = weather.loc[today,'0.precipIntensity.hourly']
		elif var == 'windSpeed_hour_3' :
			df.loc[n,var] = weather.loc[today,'3.windSpeed.hourly']
		elif var == 'windBearing_hour_4' :
			df.loc[n,var] = weather.loc[today,'4.windBearing.hourly']
	for var in newcols_yest :
				if var == '1_day_prior_dewPoint':
					df.loc[n,var] = weather.loc[yesterday,'dewPoint']
				elif var == '1_day_prior_presure':
					df.loc[n,var] = weather.loc[yesterday,'pressure']
				elif var == 'cloudCover_hour_-15':
					df.loc[n,var] = weather.loc[yesterday,'9.cloudCover.hourly']
				elif var == 'temperature_hour_-19':
					df.loc[n,var] = weather.loc[yesterday,'5.temperature.hourly']
				elif var == 'temperature_hour_-18':
					df.loc[n,var] = weather.loc[yesterday,'6.temperature.hourly']
				elif var == 'temperature_hour_-17':
					df.loc[n,var] = weather.loc[yesterday,'7.temperature.hourly']
				elif var == 'temperature_hour_-16':
					df.loc[n,var] = weather.loc[yesterday,'8.temperature.hourly']
				elif var == 'temperature_hour_-15':
					df.loc[n,var] = weather.loc[yesterday,'9.temperature.hourly']
				elif var == 'temperature_hour_-14':
					df.loc[n,var] = weather.loc[yesterday,'10.temperature.hourly']
				elif var == 'temperature_hour_-13':
					df.loc[n,var] = weather.loc[yesterday,'11.temperature.hourly']
				elif var == 'temperature_hour_-12':
					df.loc[n,var] = weather.loc[yesterday,'12.temperature.hourly']
				elif var == 'temperature_hour_-11':
					df.loc[n,var] = weather.loc[yesterday,'13.temperature.hourly']
				elif var == 'temperature_hour_-10':
					df.loc[n,var] = weather.loc[yesterday,'14.temperature.hourly']
				elif var == 'temperature_hour_-9':
					df.loc[n,var] = weather.loc[yesterday,'15.temperature.hourly']
				elif var == 'temperature_hour_-8':
					df.loc[n,var] = weather.loc[yesterday,'16.temperature.hourly']
				elif var == 'temperature_hour_-7':
					df.loc[n,var] = weather.loc[yesterday,'17.temperature.hourly']
				elif var == 'temperature_hour_-6':
					df.loc[n,var] = weather.loc[yesterday,'18.temperature.hourly']
				elif var == 'temperature_hour_-5':
					df.loc[n,var] = weather.loc[yesterday,'19.temperature.hourly']
				elif var == 'temperature_hour_-4':
					df.loc[n,var] = weather.loc[yesterday,'20.temperature.hourly']
				elif var == 'temperature_hour_-3':
					df.loc[n,var] = weather.loc[yesterday,'21.temperature.hourly']
				elif var == 'temperature_hour_-2':
					df.loc[n,var] = weather.loc[yesterday,'22.temperature.hourly']
				elif var == 'temperature_hour_-1':
					df.loc[n,var] = weather.loc[yesterday,'23.temperature.hourly']
				elif var == 'windVectorX_hour_-19':
					df.loc[n,var]=np.cos(np.pi*2*weather.loc[yesterday,'5.windBearing.hourly']/360)*weather.loc[yesterday,'5.windSpeed.hourly']
				elif var == 'windVectorX_hour_-14':
					df.loc[n,var]=np.cos(np.pi*2*weather.loc[yesterday,'10.windBearing.hourly']/360)*weather.loc[yesterday,'10.windSpeed.hourly']
				elif var == 'windVectorX_hour_-9':
					df.loc[n,var]=np.cos(np.pi*2*weather.loc[yesterday,'15.windBearing.hourly']/360)*weather.loc[yesterday,'15.windSpeed.hourly']
				elif var == 'windVectorX_hour_-5':
					df.loc[n,var]=np.cos(np.pi*2*weather.loc[yesterday,'19.windBearing.hourly']/360)*weather.loc[yesterday,'19.windSpeed.hourly']
				elif var == 'windVectorY_hour_-19':
					df.loc[n,var]=np.sin(np.pi*2*weather.loc[yesterday,'5.windBearing.hourly']/360)*weather.loc[yesterday,'5.windSpeed.hourly']
				elif var == 'windVectorY_hour_-14':
					df.loc[n,var]=np.sin(np.pi*2*weather.loc[yesterday,'10.windBearing.hourly']/360)*weather.loc[yesterday,'10.windSpeed.hourly']
				elif var == 'windVectorY_hour_-9':
					df.loc[n,var]=np.sin(np.pi*2*weather.loc[yesterday,'15.windBearing.hourly']/360)*weather.loc[yesterday,'15.windSpeed.hourly']
				elif var == 'windVectorY_hour_-5':
					df.loc[n,var]=np.sin(np.pi*2*weather.loc[yesterday,'19.windBearing.hourly']/360)*weather.loc[yesterday,'19.windSpeed.hourly']
				elif var == 'pressure_hour_-8' :
					df.loc[n,var] = weather.loc[yesterday,'16.pressure.hourly']
				elif var == 'Max_precipIntensity-1':
					df.loc[n,var] = weather.loc[yesterday,'precipIntensityMax']
				elif var == '1_day_prior_temperatureMax':
					df.loc[n,var] = weather.loc[yesterday,'temperatureMax']
				elif var == '1_day_prior_pressure':
					df.loc[n,var] = weather.loc[yesterday,'pressure']
				elif var == 'windSpeed' :
					df.loc[n,var] = weather.loc[yesterday,'windSpeed']
				elif var == 'humidity' :
					df.loc[n,var] = weather.loc[yesterday,'humidity']
				elif var == '1_day_prior_temperatureMin' :
					df.loc[n,var] = weather.loc[yesterday,'temperatureMin']
				elif var == 'cloudCover' :
					df.loc[n,var] = weather.loc[yesterday,'cloudCover']						
	for var in newcols_twodays :
		if var == '2_day_prior_dewPoint':
			df.loc[n,var] = weather.loc[two_days,'dewPoint']
		elif var == '2_day_prior_pressure':
			df.loc[n,var] = weather.loc[two_days,'pressure']
		elif var == '2_day_prior_temperatureMax':
			df.loc[n,var] = weather.loc[two_days,'temperatureMax']
		elif var == '2_day_prior_windVectorX' :
			df.loc[n,'2_day_prior_windVectorX'] = np.cos(np.pi*2*weather.loc[two_days,'windBearing']/360)*weather.loc[two_days,'windSpeed']
		elif var == '2_day_prior_windVectorY' :
			df.loc[n,'2_day_prior_windVectorY'] = np.sin(np.pi*2*weather.loc[two_days,'windBearing']/360)*weather.loc[two_days,'windSpeed']
		elif var == 'Max_precipIntensity-2':
			df.loc[n,var] = weather.loc[two_days,'precipIntensityMax']
		elif var == '2_day_prior_temperatureMin':
			df.loc[n,var] = weather.loc[two_days,'temperatureMin']
			
	df.loc[n,'3_day_prior_temperatureMax'] = weather.loc[three_days,'temperatureMax'] #Max temperature from 3 days prior
	df.loc[n,'Max_precipIntensity-3'] = weather.loc[three_days,'precipIntensityMax']#Max Precipitation from 3 days prior
	df.loc[n,'3_day_prior_temperatureMin'] = weather.loc[three_days,'temperatureMin']#Min temperature from 3 days prior
	df.loc[n,'4_day_prior_temperatureMax'] = weather.loc[four_days,'temperatureMax'] #Max temperature from 4 days prior
	df.loc[n,'Max_precipIntensity-4'] = weather.loc[four_days,'precipIntensityMax']#Max Precipitation from 4 days prior
	df.loc[n,'4_day_prior_temperatureMin'] = weather.loc[four_days,'temperatureMin']#Min temperature from 4 days prior



df['12hrPressureChange'] = df['pressure_hour_4']-df['pressure_hour_-8']	#Get the Pressure change from 4am the day of and 4pm the day before the prediction date.

	####Group Means####
df['Mean']=pd.to_numeric(df['Mean']) #Change the 'Mean' variable to numeric so we can actually get the mean of the groups
daily_group_means = pd.DataFrame() #Create daily_group_means data frame
daily_group_means=df.groupby(['Full_date','Group'])['Mean'].mean() #Save the means of each group to daily_group_means. There are 2 indexes 'Full_date' and 'Group'. With the variable 'Mean' for each.

for row in range(len(df)) :
	day, grp =df.loc[row,['Full_date','Group']] #For each row in df get the date and group
	df.loc[row,'group_prior_mean']= daily_group_means.loc[(day,grp)] #Using daily_group_means pull the mean for the specific day and group and put it in the 'group_prior_mean' variable in df.

 ###Get the E.coli levels from each of the 7 previous days###
 ###We group the beaches together, then use the algorithm: Take the column from the previous day's shift and shift it by one row, and save it. The beaches are grouped together so when you shift the rows you do not get a reading from a different beach. Repeat multiple times until you have the previous 7 readings####
 
df['2_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['Mean'].shift(-1)

df['3_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['2_day_prior_Escherichia.coli'].shift(-1)

df['4_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['3_day_prior_Escherichia.coli'].shift(-1)

df['5_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['4_day_prior_Escherichia.coli'].shift(-1)

df['6_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['5_day_prior_Escherichia.coli'].shift(-1)

df['7_day_prior_Escherichia.coli'] =df.groupby(['Beach'])['6_day_prior_Escherichia.coli'].shift(-1)

df['accum_rain'] = df[['Max_precipIntensity-1','Max_precipIntensity-2','Max_precipIntensity-3','Max_precipIntensity-4']].sum(axis=1) #Creates the 'accum_rain' variable from summing how much rain happened over the past 4 days

df['trailing_average_daily_temperatureMax'] = df[['1_day_prior_temperatureMax','2_day_prior_temperatureMax','3_day_prior_temperatureMax','4_day_prior_temperatureMax']].mean(axis=1)#Creates the 'trailing_average_daily_temperatureMax' variable from averaging the max temperatures from the past 4 days

df['trailing_average_daily_temperatureMin'] = df[['1_day_prior_temperatureMin','2_day_prior_temperatureMin','3_day_prior_temperatureMin','4_day_prior_temperatureMin']].mean(axis=1)#Creates the 'trailing_average_daily_temperatureMax' variable from averaging the min temperatures from the past 4 days

df['trailing_average_daily_Escherichia.coli']= df[['Mean','2_day_prior_Escherichia.coli',
'3_day_prior_Escherichia.coli',
'4_day_prior_Escherichia.coli',
'5_day_prior_Escherichia.coli',
'6_day_prior_Escherichia.coli',
'7_day_prior_Escherichia.coli']].mean(axis=1)#Creates the 'trailing_average_daily_Escherichia.coli' variable from averaging the past 7 ecoli readings at a specific location

df['trailing_average_daily_pressure']= df[['1_day_prior_pressure','2_day_prior_pressure']].mean(axis=1)#Creates the 'trailing_average_daily_pressure' variable from averaging the pressures from the past 2 days

df['trailing_average_daily_dewPoint']= df[['1_day_prior_dewPoint','2_day_prior_dewPoint']].mean(axis=1)#Creates the 'trailing_average_daily_dewPoint' variable from averaging the dew points from the past 2 days

df['trailing_average_hourly_windVectorX'] = df[['windVectorX_hour_4','windVectorX_hour_0','windVectorX_hour_-5','windVectorX_hour_-9','windVectorX_hour_-14','windVectorX_hour_-19']].mean(axis=1)#Creates the 'trailing_average_hourly_windVectorX' variable from averaging the wind vectors in the X direction from the day before the prediction date.

df['trailing_average_hourly_windVectorY'] = df[['windVectorY_hour_4','windVectorY_hour_0','windVectorY_hour_-5','windVectorY_hour_-9','windVectorY_hour_-14','windVectorY_hour_-19']].mean(axis=1)#Creates the 'trailing_average_hourly_windVectorY' variable from averaging the wind vectors in the Y direction from the day before the prediction date.

df['trailing_average_hourly_temperature'] = df[['temperature_hour_4', 'temperature_hour_0', 'temperature_hour_-5','temperature_hour_-9', 'temperature_hour_-14', 'temperature_hour_-19']].mean(axis=1)#Creates the 'trailing_average_hourly_temperature' variable from averaging the temperature from the day before the prediction date.

model_cols = ('Client.ID','windVectorX_hour_-5','windVectorY_hour_-9','group_prior_mean','windVectorY_hour_0','temperature_hour_4','temperature_hour_-5','temperature_hour_0', 'windVectorY_hour_4', 'accum_rain', 'categorical_beach_grouping', '12hrPressureChange', 'windVectorX_hour_0', 'temperature_hour_-19', 'windVectorX_hour_4', 'temperature_hour_-14','windVectorX_hour_-14', 'previous_reading','cloudCover_hour_-15', 'humidity_hour_4', 'windVectorX_hour_-9','windVectorY_hour_-19','windVectorY_hour_-5', 'Collection_Time', 'windVectorX_hour_-19', 'pressure_hour_0', 'temperature_hour_-9', 'windVectorY_hour_-14','2_day_prior_Escherichia.coli', '3_day_prior_Escherichia.coli', '4_day_prior_Escherichia.coli', '5_day_prior_Escherichia.coli', '6_day_prior_Escherichia.coli', '7_day_prior_Escherichia.coli', '2_day_prior_temperatureMax', '3_day_prior_temperatureMax', '4_day_prior_temperatureMax','2_day_prior_windVectorX', '2_day_prior_windVectorY','1_day_prior_pressure', '2_day_prior_pressure', '1_day_prior_dewPoint', '2_day_prior_dewPoint','trailing_average_daily_Escherichia.coli','trailing_average_daily_temperatureMax','trailing_average_daily_pressure', 'trailing_average_daily_dewPoint', 'trailing_average_hourly_temperature','trailing_average_hourly_windVectorX','trailing_average_hourly_windVectorY') #Gets the exact columns in the order that you need to run the RF and the GBM models. 

svc_cols = ('Client.ID', 'windVectorX_hour_-9', 'accum_rain', 'temperature_hour_0', 'windVectorY_hour_-9', 'categorical_beach_grouping', '12hrPressureChange', 'group_prior_mean', 'windVectorY_hour_4', 'temperature_hour_-5', 'temperature_hour_4', 'windVectorX_hour_4', 'windVectorY_hour_-5', 'previous_reading', 'windVectorY_hour_0', 'temperature_hour_-14', 'windVectorY_hour_-19', 'windVectorX_hour_-5', 'cloudCover_hour_-15', 'pressure_hour_0', 'humidity_hour_4', 'windVectorX_hour_-14', 'temperature_hour_-9', 'windVectorX_hour_0', 'Collection_Time', 'windVectorY_hour_-14', 'windVectorX_hour_-19', 'temperature_hour_-19', '1_day_prior_pressure', '2_day_prior_pressure', '2_day_prior_windVectorX', '2_day_prior_windVectorY', '2_day_prior_temperatureMax', '3_day_prior_temperatureMax', '4_day_prior_temperatureMax', '1_day_prior_dewPoint', '2_day_prior_dewPoint', '2_day_prior_Escherichia.coli', '3_day_prior_Escherichia.coli', '4_day_prior_Escherichia.coli', '5_day_prior_Escherichia.coli', '6_day_prior_Escherichia.coli', '7_day_prior_Escherichia.coli', 'trailing_average_daily_pressure', 'trailing_average_daily_temperatureMax', 'trailing_average_daily_dewPoint', 'trailing_average_daily_Escherichia.coli', 'trailing_average_hourly_windVectorX', 'trailing_average_hourly_windVectorY', 'trailing_average_hourly_temperature')  #Gets the exact columns in the order that you need to run the SVC models. 

df=df.rename(columns={'Group':'categorical_beach_grouping','Mean':'previous_reading'}) #Rename columns to match model wording

model_df = pd.DataFrame() #Create model_df dataframe
for cols in model_cols :
	model_df[cols] = df[cols]

svc_df = pd.DataFrame() #Create svc_df dataframe
for cols in model_cols :
	svc_df[cols] = df[cols]

model_df = model_df.dropna() #Drop all the rows with 'NAN's in them. Can not run models with na in any columns. 
svc_df = svc_df.dropna() 

rf_preds = pd.DataFrame() #Create rf_preds dataframe 
gbm_preds = pd.DataFrame() #Create gbm_preds dataframe
SVC_preds = pd.DataFrame()#Create svc_preds dataframe

for yr in range (2006,2015) :
	filename = ('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/RF_regress_'+str(yr)+'.pkl') # Open the RF model files

	rfmodel = joblib.load(filename) #Load the RF models

	rf_preds['rf_'+str(yr)]=getattr(rfmodel, 'predict')(model_df) #Run the RF models
	
	filename = ('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/gbm_regress_'+str(yr)+'.pkl') #Open the GBM model files

	gbmmodel = joblib.load(filename) #Load the GBM models

	gbm_preds['gbm_'+str(yr)]=getattr(gbmmodel, 'predict')(model_df) #Run the GBM models
filename = ('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/svc_clf.pkl') #Open the SVC model file

svcmodel = joblib.load(filename) #Load the SVC models
svc_df=preprocessing.scale(svc_df) #Process the data so that the variation isn't as big.

for x in range(0,len(svc_df)):
	SVC_preds.loc[x,'SVC_prediction'] = svcmodel.predict(svc_df[[x]])[0] #Run the SVC models
	

model_df['previous_reading'] =model_df['previous_reading'].shift(1) #Shift the 'previous_reading' column down a row, to show what the actual E.coli reading was on the prediction day.



report = pd.read_csv('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/ValidationReport2.csv',dtype={'RF_thresh2p':np.float64,'RF_thresh5p':np.float64,'GBM_thresh2p':np.float64,'GBM_thresh5p':np.float64}) #Read in the ValidationReport2.csv and name the columns and type from that file.

#Find the mean of thresholds from the past RF and GBM models of having 2 or 5 percent false positives.
RF_thresh2p = report.RF_thresh2p.mean(axis=0)
RF_thresh5p = report.RF_thresh5p.mean(axis=0)
GBM_thresh2p = report.GBM_thresh2p.mean(axis=0)
GBM_thresh5p = report.GBM_thresh5p.mean(axis=0)

#Create the test data frame with previous readings, RF predictions, GBM predictions, and SVC predictions
test = pd.concat([SVC_preds,model_df['previous_reading'].reset_index(),rf_preds.mean(axis=1), gbm_preds.mean(axis=1)],axis=1)

#Rename columns that are 0 and 1 to 'RF_Predictions' and 'GBM_Predictions' respectively
test = test.rename(columns={0:'RF_Predictions',1:'GBM_Predictions'})

#SVC Plot
plt.scatter(test['previous_reading'],test['SVC_prediction'])
plt.axvline(x=235,color='r')
plt.ylabel('Ecoli Level')
plt.xlabel('SVC Predictions')
plt.title('SVC Model')
plt.show()

#RF Plot
plt.scatter(test['RF_Predictions'],test['previous_reading'])
plt.axhline(y=235,color='r')
plt.ylabel('Ecoli Level')
plt.xlabel('RF Predictions')
plt.title('RF Model')
plt.show()

#GBM Plot
plt.scatter(test['GBM_Predictions'],test['previous_reading'])
plt.axhline(y=235,color='r')
plt.ylabel('Ecoli Level')
plt.xlabel('GBM Predictions')
plt.title('GBM Model')
plt.show()

SVC_threshold = 1.0
RF_threshold = 4.8
GBM_threshold = 7.01

#Non-weighted Confusion Matrix Prep
for i in range(len(test)) :
	total = 0 #Set a total variable to see how many models predicted true/false. If the model predicted true add 1 to total.
	if test.loc[i,'previous_reading'] < 235 or math.isnan(test.loc[i,'previous_reading']): #Find whether the actual prediction day E.coli levels were high or not.
		test.loc[i,'E.coli'] = False
	else:
		test.loc[i,'E.coli'] = True
	if test.loc[i,'SVC_prediction']== SVC_threshold : 
		total +=1
	if test.loc[i,'RF_Predictions']> RF_threshold: 
		total +=1
	if test.loc[i,'GBM_Predictions']> GBM_threshold:
		total +=1
	if total >=2 : #Use this to see how many models predicted true.
		test.loc[i,'Predict']=True
	else:
		test.loc[i,'Predict'] = False

pd.crosstab(test['E.coli'],test['Predict']) #Create the confusion matrix

#Weighted confusion matrix prep. This is to try and getting all the SVC/RF combos out of the democratic model. The GBM works the best with both the RF and SVC. But the RF and SVC are not compatible together. This uses basically the same algorithm as above, but instead of adding 1 for each true, a true is weighted for each model.
for i in range(len(test)) :
	total = 0
	if test.loc[i,'previous_reading'] < 235 or math.isnan(test.loc[i,'previous_reading']):
		test.loc[i,'E.coli'] = False
	else:
		test.loc[i,'E.coli'] = True
	if test.loc[i,'SVC_prediction']== 1.0 :
		total +=.05
	if test.loc[i,'RF_Predictions']> 4.8:
		total +=.15
	if test.loc[i,'GBM_Predictions']> 7.01:
		total +=.8
	if total >= .81 :
		test.loc[i,'Predict']=True
	else:
		test.loc[i,'Predict'] = False



pd.crosstab(test['E.coli'],test['Predict'])

