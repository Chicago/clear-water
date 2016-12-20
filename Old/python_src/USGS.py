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
	#Open City of Chicago connection
	resp = requests.get('https://data.cityofchicago.org/api/views/2ivx-z93u/rows.json?accessType=DOWNLOAD')
	#Put the Data from Chicago into a theJSON variable
	theJSON = json.loads(resp.text)
	
	data = []
	
	#Grab (Time stamp, Beach, Sample 1&2, and Mean columns from city website.)
	for item in theJSON["data"]:
		data.append(item[9:14]) 
		
	#The table df will be the main data frame to store all the variables
	#for before doing the final models.
	df_data=pd.DataFrame(data) #Save the data we just got from City of Chicago

	#Rename first 5 columns in df
	df_data.columns=['Full_date','Beach','Reading1','Reading2','Mean']
	return df_data


######################################################
#### Read in USGS predicted readings from the city website
######################################################
def USGS_predict() :
	#Open City of Chicago connection
	resp = requests.get('https://data.cityofchicago.org/api/views/t62e-8nvc/rows.json?accessType=DOWNLOAD')
	#Put the Data from Chicago into a theJSON variable
	theJSON = json.loads(resp.text)
	
	USGS = []

	#Grab (`Beach`, `TimeStamp`, `Predicted Level`, `Probability`,
	#and `Swim Advisory` columns from city website.)
	for item in theJSON["data"]:
		USGS.append(item[8:13])

	#Save the data we just got from City of Chicago to a pandas data frame.
	df_USGS = pd.DataFrame(USGS)

	 #Rename first 5 columns in df
	df_USGS.columns = ['Beach', 'Full_date','Predicted_Level','Probability', 'Swim_Advisory']
	return df_USGS
	
df_data= Ecoli_Readings()
df_USGS= USGS_predict()

first_day = '2016-01-01'
last_day = '2016-12-31'

#Parse 'Full_date' in first column to keep just the date. 
df_data['Full_date']=df_data['Full_date'].str.extract('(....-..-..)',expand=False)

#Parse 'Full_date' in first column to keep just the date. 
df_USGS['Full_date']=df_USGS['Full_date'].str.extract('(....-..-..)',expand=False) 

#Sort by 'Beach' and 'Full_date' to organize data better.
df_data = df_data.sort_values(by=['Beach','Full_date'],ascending=[True,False])

#Sort by 'Beach' and 'Full_date' to organize data better.
df_USGS = df_USGS.sort_values(by=['Beach','Full_date'],ascending=[True,False])

#Trim df_data down to the days we are going to be looking for at each beach.
df_data = df_data[(df_data.Full_date >= first_day)&(df_data.Full_date <= last_day)]

#Trim df_USGS down to the days we are going to be looking for at each beach.
df_USGS = df_USGS[(df_USGS.Full_date >= first_day)&(df_USGS.Full_date <= last_day)]

#The index is of the rows where the days we are analyzing were pulled from,
# before we trimmed df down. So we reset the whole index to just give us the
# rows that we are using.
df_data = df_data.reset_index(drop=True)


#The index is of the rows where the days we are analyzing were pulled from,
#before we trimmed df down. If this is not done, the following for loop will
#not be able to run.
df_USGS = df_USGS.reset_index(drop=True)

#Open the Current Working Directory
cwd = os.getcwd()

#Using the Beaches_LatLong.csv we read in all that data, for the ability to 
#standardize the 'Beach' variable.
latlong = pd.read_csv(cwd+'\\Beaches_LatLong.csv',dtype={'Client.ID':object,
'Latitude':str,'Longitude':str,'Group':str,'North':str,'USGS':str})


#Rename the beaches in both df_data and df_USGS so they match each other.
#Also create a `USGS_id` to identify the beaches that use the same predictions
#from the USGS model
for i in range(len(df_data['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df_data.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df_data.loc[i,'Beach']= latlong.loc[j,'Beach']
			df_data.loc[i,'USGS_id']= latlong.loc[j,'USGS']
			
for i in range(len(df_USGS['Beach'])) :
	for j in range(len(latlong['Online'])) :
		if df_USGS.loc[i,'Beach'] == latlong.loc[j,'Online'] :
			df_USGS.loc[i,'Beach']= latlong.loc[j,'Beach']
			df_USGS.loc[i,'USGS_id']= latlong.loc[j,'USGS']
			
#Generate a data frame		
full_data = pd.merge(df_USGS,df_data, on=['USGS_id','Full_date'], how = 'inner')

partial_USGS = pd.merge(df_USGS,df_data,on=['Beach','Full_date'], how = 'inner')

for i in range(len(full_data)) :
	#Set a total variable to see how many models predicted true/false. 
	#If the model predicted true add 1 to total.
	total = 0
	#Find whether the actual prediction day E.coli levels were high or not.
	if pd.to_numeric(full_data.loc[i,'Mean']) < 235: 
		full_data.loc[i,'E.coli'] = False
	else:
		full_data.loc[i,'E.coli'] = True
	#If A swim ban was in place due to the USGS model.
	if full_data.loc[i,'Swim_Advisory'] == 'N' :
		full_data.loc[i,'Predict']= False
	else :
		full_data.loc[i,'Predict']= True

		
for i in range(len(partial_USGS)) :
	#Set a total variable to see how many models predicted true/false. 
	#If the model predicted true add 1 to total.
	total = 0
	#Find whether the actual prediction day E.coli levels were high or not.
	if pd.to_numeric(partial_USGS.loc[i,'Mean']) < 235: 
		partial_USGS.loc[i,'E.coli'] = False
	else:
		partial_USGS.loc[i,'E.coli'] = True
	#If A swim ban was in place due to the USGS model.
	if partial_USGS.loc[i,'Swim_Advisory'] == 'N' :
		partial_USGS.loc[i,'Predict']= False
	else :
		partial_USGS.loc[i,'Predict']= True

####Create the confusion matrix####
print('Full USGS predictions\n',pd.crosstab(full_data['E.coli'],full_data['Predict'])) 

print('Partial USGS predictions\n',pd.crosstab(partial_USGS['E.coli'],partial_USGS['Predict'])) 
