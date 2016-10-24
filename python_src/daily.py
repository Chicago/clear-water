import requests
import pandas as pd
import json
import re
import read_data as rd
import math
import numpy as np
import daily_weather as dw
import datetime
import os
from sklearn.externals import joblib
'''
######################################################
####DataFrames used in this program
######################################################

#df - The main Data frame to which all necessary variables will be stored

#temp - A data frame to store the most recent E.coli readings to
		transfer to df.

#group_means - A data set that is a collection of the means for E.coli of
				the 7 groups of beaches

#model_df - The data frame we run the RF and GBM models from.

#rf_preds - Data Frame to hold the RF Predictions. This will have the
			predictions for each RF model from build_models.py

#gbm_preds - Data Frame to hold the GBM Predictions. This will have the 
			predictions for each GBM model from build_models.py 

#predict - The data frame with all the final predictions for the next day.
			
'''

######################################################
#### Get a DarkSky.net API that works
######################################################
status = False
while status == False:
	apikey= str(input('Enter your Dark Sky APIkey:'))
	url = 'https://api.forecast.io/forecast/'+apikey+'/41.878311,-87.616342,'+datetime.date.today().isoformat()+'T00:00:00'
	r = requests.get(url)
	status = r.status_code == requests.codes.ok
	
	
######################################################
#### Read in data from the city website
######################################################

#Open City of Chicago connection
resp = requests.get('https://data.cityofchicago.org/api/views/2ivx-z93u/rows.json?accessType=DOWNLOAD')

#Put the Data from Chicago into a theJSON variable
theJSON = json.loads(resp.text) 
data = []

#Grab (Timestamp, Beach, Sample 1&2, and Mean) columns from city website.

for item in theJSON["data"]:
	data.append(item[9:14]) 

#The table df will be the main data frame to store all the variables
#for before doing the final models.

df=pd.DataFrame(data) 

#Split apart the timestamp column
for d in range(len(df)):
	ts=datetime.datetime.strptime(df.iloc[d,0],'%Y-%m-%dT%H:%M:%S')
	#The 'Collection time' in all the models are in minutes after midnight.
	df.loc[d,'Collection_Time'] = int(ts.strftime('%H'))*60+int(ts.strftime('%M'))

	
######################################################
####Populating the main DataFrame, df
######################################################
#Create 'Full_date' in first column		
df[0]=df[0].str.extract('(....-..-..)',expand=False)  

#Rename first 6 columns in df
df.columns=['Full_date','Beach','Reading1','Reading2','Mean','Collection_Time'] 

#Sort by 'Beach' and 'Full_date' to organize data better.
df= df.sort_values(by=['Beach','Full_date'],ascending=[True,False])

#Get rid of all the data from beaches that we do not analyze. 
df= df.loc[(df.Beach != "Columbia")&(df.Beach!="Lane")&(df.Beach!="Loyola")&(df.Beach!="Marion Mahoney Griffin")&(df.Beach!="North Shore")&(df.Beach!="NA")]

######################################################
####Put the most recent E.coli levels into df
######################################################
#Create a temporary data frame to grab the most recent E.coli readings.	
temp = df

#Group by all the beaches, and keep only the latest reading from each beach
df=df.groupby('Beach').first() 

#Not going to need the rest of the list of readings, the index is reset.
df.reset_index(level=0,inplace=True) 

days_prior_ecoli = 8 #How many days prior we want the readings for

#For each beach grab how many days prior reading,
#and save if to the temp data frame
temp = temp.groupby('Beach').head(days_prior_ecoli).reset_index(drop=True)

#Create a column for E. Coli readings for each one of the previous days
for number in range(days_prior_ecoli):
	df['%d_day_prior_Escherichia.coli'%(number)]=np.nan 

i=0 #This will keep track of the row we are working on in the temp data frame

for beaches in range(math.ceil(len(temp)/days_prior_ecoli)) :
	for days in range(days_prior_ecoli):
		df.loc[beaches,'%d_day_prior_Escherichia.coli'%(days)]=float(temp.loc[i,'Mean']) #Transfer the info in the temp data frame to the df data frame
		i+=1 #go to the next row in the temp data frame
df =df.drop('0_day_prior_Escherichia.coli',axis=1) # this is the same as the 'previous_reading' variable in df so take it out

######################################################
####Add the weather to df
######################################################	

#LatLong is a csv with the basic information for each beach
df['Client.ID']=np.nan 
df['lat']=np.nan
df['long']=np.nan
df['Group']=np.nan
df['North']=np.nan

#What type of information will be stored in each column
latlong = pd.read_csv('Beaches_LatLong.csv',dtype={'Client.ID':object,'Latitude':str,'Longitude':str,'Group':str,'North':str}) 

#Take all the info in the LatLong.csv and put it to the df data frame.
for i in range(len(df['Beach'])) :
	for j in range(len(latlong['Online'])) :
	#Find the specific beach for each row to add the information.
		if df.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df.loc[i,'Beach']= latlong.loc[j,'Beach']
			df.loc[i,'lat'] = latlong.loc[j,'Latitude']
			df.loc[i,'long'] = latlong.loc[j,'Longitude']
			df.loc[i,'Client.ID']= latlong.loc[j,'Client.ID']
			df.loc[i,'Group'] = latlong.loc[j,'Group']
			df.loc[i,'North'] = latlong.loc[j,'North']
	
#These are going to be the columns of weather for the specific day that we
# are predicting in the models.
newcols_today = ('humidity_hour_4','pressure_hour_0','temperature_hour_0',
				'temperature_hour_4','windVectorX_hour_0',
				'windVectorX_hour_4','windVectorY_hour_0',
				'windVectorY_hour_4','pressure_hour_4','temperature_hour_1',
				'temperature_hour_2','temperature_hour_3','windSpeed_hour_4',
				'precipIntensity_hour_0','windBearing_hour_4') 

for var in newcols_today :
	df[var]=np.nan


#These are going to be the columns of weather for the day before that
# we are predicting in the models.	
newcols_yest = ('1_day_prior_dewPoint','1_day_prior_pressure',
				'cloudCover_hour_-15','temperature_hour_-19',
				'temperature_hour_-19','temperature_hour_-18',
				'temperature_hour_-17','temperature_hour_-16',
				'temperature_hour_-15','temperature_hour_-14',
				'temperature_hour_-13','temperature_hour_-12',
				'temperature_hour_-11','temperature_hour_-10',
				'temperature_hour_-9','temperature_hour_-8',
				'temperature_hour_-7','temperature_hour_-6',
				'temperature_hour_-5','temperature_hour_-4',
				'temperature_hour_-3','temperature_hour_-2',
				'temperature_hour_-1','windVectorX_hour_-19',
				'windVectorX_hour_-14','windVectorX_hour_-9',
				'windVectorX_hour_-5', 'windVectorY_hour_-19',
				'windVectorY_hour_-14','windVectorY_hour_-9',
				'windVectorY_hour_-5','pressure_hour_-8',
				'Max_precipIntensity-1','1_day_prior_temperatureMax',
				'windSpeed','humidity','1_day_prior_temperatureMin',
				'cloudCover') 

for var in newcols_yest :
	df[var]=np.nan

	
#These are going to be the columns of weather for the two days before the day
# that we are predicting in the models.
newcols_twodays = ('2_day_prior_dewPoint', '2_day_prior_pressure', 
					'2_day_prior_temperatureMax','2_day_prior_windVectorX',
					'2_day_prior_windVectorY','Max_precipIntensity-2',
					'2_day_prior_temperatureMin')

for var in newcols_twodays :
	df[var]=np.nan
	
df['3_day_prior_temperatureMax']= np.nan
df['4_day_prior_temperatureMax']= np.nan

#Go through each row of df and add in the information for all the columns
# we created above and populate the dataFrame
for i in range(len(df)) : #for each row/beach we are predicting for
	#Grab the Latitude of the beach we are going to be pulling the weather for
	lattitude = str(df.loc[i,'lat'])
	#Grab the Latitude of the beach we are going to be pulling the weather for
	longitude = str(df.loc[i,'long']) 
	for day in range(5) :#for each of the past 5 days at the beach/row.
		if day == 0:
			#Find the day we are grabbing the weather for
			d = pd.to_datetime(df.loc[i,'Full_date'],format = '%Y-%m-%d')+datetime.timedelta(days=day+1) 
		else :
			#Find the day we are grabbing the weather for
			d = pd.to_datetime(df.loc[i,'Full_date'],format = '%Y-%m-%d')-datetime.timedelta(days=day-1)
		#Strip the date and time we found for d
		d = datetime.datetime.strptime(str(d),'%Y-%m-%d %H:%M:%S')

		#Grab the weather for the day
		weather = dw.daily_weather(lat= lattitude, long=longitude,date=d.strftime("%Y-%m-%dT%H:%M:%S"),apikey=apikey)
		
		#Fill in the weather variables needed for the day that you are predicting
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
		#Fill in the weather variables needed for the day before that
		#you are predicting
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
		#Fill in the weather variables needed for 2 days before prediction date
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
		#Fill in the weather variables needed for 3 days before prediction date
		elif day == 3 : 
			df.loc[i,'3_day_prior_temperatureMax'] = weather.loc[0,'temperatureMax']
			df.loc[i,'Max_precipIntensity-3'] = weather.loc[0,'precipIntensityMax']
			df.loc[i,'3_day_prior_temperatureMin'] = weather.loc[0,'temperatureMin']
		#Fill in the weather variables needed for 4 days before prediction date
		elif day == 4 :
			df.loc[i,'4_day_prior_temperatureMax'] = weather.loc[0,'temperatureMax'] 
			df.loc[i,'Max_precipIntensity-4'] = weather.loc[0,'precipIntensityMax']
			df.loc[i,'4_day_prior_temperatureMin'] = weather.loc[0,'temperatureMin'] 


#The change in pressure from 4pm the day before to 4am the day of prediction.			
df['12hrPressureChange'] = df['pressure_hour_4']-df['pressure_hour_-8']			

#Create the group means Data Frame
group_means = pd.DataFrame()

#Create a variable that has 1-6 to save the means to that row.
group_means['Group']=range(1,7)

#Change the 'Mean' variable to a float numeric type
df['Mean'] = pd.to_numeric(df['Mean'])

#Find the Mean of each group at the prior reading
for i in range(int(df['Group'].min())-1,int(df['Group'].max())) :
	group_means.loc[i,'group_prior_mean']=df.groupby('Group')['Mean'].mean()[i]
	
for row in range(len(df['Group'])) :
	df.loc[row,'group_prior_mean'] = group_means.loc[int(df.loc[row,'Group'])-1,'group_prior_mean'] 

#For each row in DF get add all of the 'Max_precip' for the past few days.
df['accum_rain'] = df[['Max_precipIntensity-1','Max_precipIntensity-2','Max_precipIntensity-3','Max_precipIntensity-4']].sum(axis=1)

#Average the high temperatures from the past 4 days
df['trailing_average_daily_temperatureMax'] = df[['1_day_prior_temperatureMax','2_day_prior_temperatureMax','3_day_prior_temperatureMax','4_day_prior_temperatureMax']].mean(axis=1)

#Average the low temperatures from the past 4 days
df['trailing_average_daily_temperatureMin'] = df[['1_day_prior_temperatureMin','2_day_prior_temperatureMin','3_day_prior_temperatureMin','4_day_prior_temperatureMin']].mean(axis=1)

#Creates the 'trailing_average_daily_Escherichia.coli' variable from
# averaging the past 7 ecoli readings at a specific location
df['trailing_average_daily_Escherichia.coli']= df[['1_day_prior_Escherichia.coli','2_day_prior_Escherichia.coli',
'3_day_prior_Escherichia.coli',
'4_day_prior_Escherichia.coli',
'5_day_prior_Escherichia.coli',
'6_day_prior_Escherichia.coli',
'7_day_prior_Escherichia.coli']].mean(axis=1)

#Creates the 'trailing_average_daily_pressure' variable from averaging
# the pressures from the past 2 days
df['trailing_average_daily_pressure']= df[['1_day_prior_pressure','2_day_prior_pressure']].mean(axis=1)

#Creates the 'trailing_average_daily_dewPoint' variable from averaging
#the dew points from the past 2 days
df['trailing_average_daily_dewPoint']= df[['1_day_prior_dewPoint','2_day_prior_dewPoint']].mean(axis=1)

#Creates the 'trailing_average_hourly_windVectorX' variable from averaging
#the wind vectors in the X direction from the day before the prediction date.
df['trailing_average_hourly_windVectorX'] = df[['windVectorX_hour_4','windVectorX_hour_0','windVectorX_hour_-5','windVectorX_hour_-9','windVectorX_hour_-14','windVectorX_hour_-19']].mean(axis=1) 

#Creates the 'trailing_average_hourly_windVectorY' variable from averaging
# the wind vectors in the Y direction from the day before the prediction date.
df['trailing_average_hourly_windVectorY'] = df[['windVectorY_hour_4','windVectorY_hour_0','windVectorY_hour_-5','windVectorY_hour_-9','windVectorY_hour_-14','windVectorY_hour_-19']].mean(axis=1)

#Creates the 'trailing_average_hourly_temperature' variable from averaging the
#temperature from the day before the prediction date.
df['trailing_average_hourly_temperature'] = df[['temperature_hour_4', 'temperature_hour_0', 'temperature_hour_-5','temperature_hour_-9', 'temperature_hour_-14', 'temperature_hour_-19']].mean(axis=1) 

######################################################
####Creating the Data Frame to make predictions on
######################################################	

#Get the exact columns and order that you need to run the RF and the GBM models.
model_cols = ('Client.ID','windVectorX_hour_-5','windVectorY_hour_-9',
				'group_prior_mean','windVectorY_hour_0','temperature_hour_4',
				'temperature_hour_-5','temperature_hour_0','windVectorY_hour_4',
				'accum_rain', 'categorical_beach_grouping','12hrPressureChange',
				'windVectorX_hour_0', 'temperature_hour_-19',
				'windVectorX_hour_4', 'temperature_hour_-14',
				'windVectorX_hour_-14', 'previous_reading',
				'cloudCover_hour_-15', 'humidity_hour_4',
				'windVectorX_hour_-9','windVectorY_hour_-19',
				'windVectorY_hour_-5', 'Collection_Time','windVectorX_hour_-19',
				'pressure_hour_0', 'temperature_hour_-9','windVectorY_hour_-14',
				'2_day_prior_Escherichia.coli', '3_day_prior_Escherichia.coli',
				'4_day_prior_Escherichia.coli', '5_day_prior_Escherichia.coli',
				'6_day_prior_Escherichia.coli', '7_day_prior_Escherichia.coli',
				'2_day_prior_temperatureMax', '3_day_prior_temperatureMax',
				'4_day_prior_temperatureMax','2_day_prior_windVectorX',
				'2_day_prior_windVectorY','1_day_prior_pressure',
				'2_day_prior_pressure', '1_day_prior_dewPoint',
				'2_day_prior_dewPoint','trailing_average_daily_Escherichia.coli',
				'trailing_average_daily_temperatureMax',
				'trailing_average_daily_pressure',
				'trailing_average_daily_dewPoint',
				'trailing_average_hourly_temperature',
				'trailing_average_hourly_windVectorX',
				'trailing_average_hourly_windVectorY')

#Gets the exact columns in the order that you need to run the SVC models. 
svc_cols = ('Client.ID', 'windVectorX_hour_-9', 'accum_rain',
			'temperature_hour_0', 'windVectorY_hour_-9',
			'categorical_beach_grouping', '12hrPressureChange',
			'group_prior_mean', 'windVectorY_hour_4', 
			'temperature_hour_-5', 'temperature_hour_4', 'windVectorX_hour_4',
			'windVectorY_hour_-5', 'previous_reading', 'windVectorY_hour_0', 
			'temperature_hour_-14', 'windVectorY_hour_-19', 
			'windVectorX_hour_-5', 'cloudCover_hour_-15', 'pressure_hour_0',
			'humidity_hour_4', 'windVectorX_hour_-14', 'temperature_hour_-9',
			'windVectorX_hour_0', 'Collection_Time', 'windVectorY_hour_-14',
			'windVectorX_hour_-19', 'temperature_hour_-19', 
			'1_day_prior_pressure', '2_day_prior_pressure', 
			'2_day_prior_windVectorX', '2_day_prior_windVectorY', 
			'2_day_prior_temperatureMax', '3_day_prior_temperatureMax', 
			'4_day_prior_temperatureMax', '1_day_prior_dewPoint', 
			'2_day_prior_dewPoint', '2_day_prior_Escherichia.coli', 
			'3_day_prior_Escherichia.coli', '4_day_prior_Escherichia.coli', 
			'5_day_prior_Escherichia.coli', '6_day_prior_Escherichia.coli', 
			'7_day_prior_Escherichia.coli', 'trailing_average_daily_pressure',
			'trailing_average_daily_temperatureMax',
			'trailing_average_daily_dewPoint', 
			'trailing_average_daily_Escherichia.coli', 
			'trailing_average_hourly_windVectorX', 
			'trailing_average_hourly_windVectorY', 
			'trailing_average_hourly_temperature') 


#Rename columns to match model wording
df=df.rename(columns={'Group':'categorical_beach_grouping','1_day_prior_Escherichia.coli':'previous_reading'})


model_df = pd.DataFrame() #Create model_df dataframe
for cols in model_cols :
	model_df[cols] = df[cols]#Add all the columns to the model data frame

#Take the beaches we don't care about out of the model
model_df = model_df[pd.notnull(model_df['Client.ID'])]

#Build rf_data frame for the holding of the RF predictions
rf_preds = pd.DataFrame()

#Add in the beach names to the 'rf_preds'
rf_preds['Beach'] = df.Beach[pd.notnull(df['Client.ID'])] 

#Build gbm_data frame for the holding of the GBM predictions
gbm_preds = pd.DataFrame()
#Add in the beach names to the 'gbm_preds'
gbm_preds['Beach'] = df.Beach[pd.notnull(df['Client.ID'])]
cwd = os.getcwd() #open the Current Working Directory

######################################################
############## Models
######################################################	

for yr in range (2006,2015) :
	#Open the RF .pkl files
	filename = (cwd+'\\ensemble_models\\models\\RF_regress_'+str(yr)+'.pkl')

	rfmodel = joblib.load(filename) #Load the RF models

	#RF prediction model
	rf_preds['rf_'+str(yr)]=np.exp(getattr(rfmodel, 'predict')(model_df))
	
	#Open the GBM .pkl files
	filename = (cwd+'\\ensemble_models\\models\\gbm_regress_'+str(yr)+'.pkl')

	gbmmodel = joblib.load(filename) #Load the GBM models

	#GBM prediction model
	gbm_preds['gbm_'+str(yr)]=np.exp(getattr(gbmmodel, 'predict')(model_df))

#Read in the ValidationReport2.csv and name the columns and type from that file.
report = pd.read_csv('C:/Users/Callin/Documents/GitHub/Chicago/e-coli-beach-predictions/python_src/ensemble_models/models/ValidationReport2.csv',dtype={'RF_thresh2p':np.float64,'RF_thresh5p':np.float64,'GBM_thresh2p':np.float64,'GBM_thresh5p':np.float64})

#Create predict Data Frame
predict = pd.DataFrame()
predict['Beach']=df.Beach[pd.notnull(df['Client.ID'])]
predict['RF_Predictions']=rf_preds.mean(axis=1)
predict['RF_thresh2p']=report.RF_thresh2p.mean(axis=0)
predict['RF_thresh5p']=report.RF_thresh5p.mean(axis=0)
predict['GBM_Predictions']=gbm_preds.mean(axis=1)
predict['GBM_thresh2p']=report.GBM_thresh2p.mean(axis=0)
predict['GBM_thresh5p']=report.GBM_thresh5p.mean(axis=0)

predict= predict.sort_values(by='Beach') #Sort by 'Beach' 

