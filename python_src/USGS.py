import requests
import json
import datetime
import pandas as pd
import os
import numpy as np

######################################################
#### Read in actual readings from the city website
######################################################
def Ecoli_Readings() :
	resp = requests.get('https://data.cityofchicago.org/api/views/2ivx-z93u/rows.json?accessType=DOWNLOAD') #Open City of Chicago connection
	theJSON = json.loads(resp.text) #Put the Data from Chicago into a theJSON variable
	data = []
	for item in theJSON["data"]:
		data.append(item[9:14]) #Grab (Timestamp, Beach, Sample 1&2, and Mean columns from city website.)
		
	#The table df will be the main data frame to store all the variables for before doing the final models.
	df_data=pd.DataFrame(data) #Save the data we just got from City of Chicago to the df_data data frame

	df_data.columns=['Full_date','Beach','Reading1','Reading2','Mean']#Rename first 5 columns in df
	return df_data


######################################################
#### Read in USGS predicted readings from the city website
######################################################
def USGS_predict() :
	resp = requests.get('https://data.cityofchicago.org/api/views/t62e-8nvc/rows.json?accessType=DOWNLOAD') #Open City of Chicago connection
	theJSON = json.loads(resp.text) #Put the Data from Chicago into a theJSON variable
	USGS = []

	for item in theJSON["data"]:
		USGS.append(item[8:13]) #Grab (Beach, TimeStamp, Predicte Level, Probability, and Swim Advisory columns from city website.)

	df_USGS = pd.DataFrame(USGS) #Save the data we just got from City of Chicago to a pandas data frame.

	df_USGS.columns = ['Beach', 'Full_date','Predicted_Level','Probability', 'Swim_Advisory'] #Rename first 5 columns in df
	return df_USGS
	
df_data= Ecoli_Readings()
df_USGS= USGS_predict()

first_day = '2016-01-01'
last_day = '2016-12-31'

df_data['Full_date']=df_data['Full_date'].str.extract('(....-..-..)',expand=False) #Parse 'Full_date' in first column to keep just the date. 

df_USGS['Full_date']=df_USGS['Full_date'].str.extract('(....-..-..)',expand=False) #Parse 'Full_date' in first column to keep just the date. 

df_data = df_data.sort_values(by=['Beach','Full_date'],ascending=[True,False]) #Sort by 'Beach' and 'Full_date' to organize data better.

df_USGS = df_USGS.sort_values(by=['Beach','Full_date'],ascending=[True,False]) #Sort by 'Beach' and 'Full_date' to organize data better.

df_data = df_data[(df_data.Full_date >= first_day)&(df_data.Full_date <= last_day)] #Trim df_data down to the days we are going to be looking for at each beach.

df_USGS = df_USGS[(df_USGS.Full_date >= first_day)&(df_USGS.Full_date <= last_day)] #Trim df_USGS down to the days we are going to be looking for at each beach.

df_data = df_data.reset_index(drop=True) #The index is of the rows where the days we are analyzing were pulled from, before we trimmed df down. So we reset the whole index to just give us the rows that we are using.

df_USGS = df_USGS.reset_index(drop=True) #The index is of the rows where the days we are analyzing were pulled from, before we trimmed df down. If this is not done, the following for loop will not be able to run.

cwd = os.getcwd() #open the Current Working Directory

latlong = pd.read_csv(cwd+'\\Beaches_LatLong.csv',dtype={'Client.ID':object,'Latitude':str,'Longitude':str,'Group':str,'North':str}) #Using the Beaches_LatLong.csv we read in all that data, for the 'Group' and 'North' variables.

#Rename the beaches in both df_data and df_USGS so they match each other.
for i in range(len(df_data['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df_data.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df_data.loc[i,'Beach']= latlong.loc[j,'Beach']
			
for i in range(len(df_USGS['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df_USGS.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df_USGS.loc[i,'Beach']= latlong.loc[j,'Beach']
			
# df_data = df_data.set_index(['Beach','Full_date']) #Put the index of df_data as the variables 'Beach' and 'Full_date' to make it easier to search and get the mean put on the df_USGS

# for row in range(len(df_USGS)) :
	# try :
		# day, beach =df_USGS.loc[row,['Full_date','Beach']] #For each row in df get the date and group
		# df_USGS.loc[row,'Mean']= df_data.loc[(day,beach)] #Using daily_group_means pull the mean for the specific day and group and put it in the 'group_prior_mean' variable in df.
	# except:
		# df_USGS.loc[row,'Mean'] = np.nan

		
full_data = pd.merge(df_USGS,df_data, on=['Beach','Full_date'], how = 'inner')

for i in range(len(full_data)) :
	total = 0 #Set a total variable to see how many models predicted true/false. If the model predicted true add 1 to total.
	if pd.to_numeric(full_data.loc[i,'Mean']) < 235: #Find whether the actual prediction day E.coli levels were high or not.
		full_data.loc[i,'E.coli'] = False
	else:
		full_data.loc[i,'E.coli'] = True
	if full_data.loc[i,'Swim_Advisory'] == 'N' : #If A swim ban was in place due to the USGS model.
		full_data.loc[i,'Predict']= False
	else :
		full_data.loc[i,'Predict']= True

pd.crosstab(full_data['E.coli'],full_data['Predict']) #Create the confusion matrix
