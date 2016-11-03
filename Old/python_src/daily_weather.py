#This function accesses the DarkSky.net using their API and grabs all the information that is provided for both the hourly and daily variables from the website. And returns a df with all the hourly and daily variables for a specific day.

def daily_weather(lat,long,date,apikey) :
	import requests
	import json
	import pandas as pd
	import numpy as np
	import os
	url = ('https://api.forecast.io/forecast/'+apikey+'/'+lat + ','+long +','+date) #create a string to create a url to access DarkSky.net
	resp = requests.get(url) #Get the data off of the URL
	theJSON = json.loads(resp.text) #The data comes in a json, This loads the json data into the program 

	daily_variables = []
	hourly_variables = []
	new = pd.DataFrame()
	
#Read all variables of the daily part in the json and put it to the variable daily_variables. Then put all the data for the day into the 'new' data frame.
	for key in theJSON["daily"]["data"][0]:
		daily_variables.append(key)

	for name in daily_variables:
		new[name]=np.nan

	for i in daily_variables:
		for item in theJSON["daily"]["data"]:
			new.loc[0,i] = item[i]

#Read all variables of the hourly part in the json and put it to the variable hourly_variables 		
	for key in theJSON["hourly"]["data"][0]:
		hourly_variables.append(key)

#For each one of the variables in the json, create a column for each hour from 0(midnight)-23(11p.m.)
	for name in hourly_variables:
		for j in range(24):
			new[str(j)+"."+name+".hourly"]=np.nan
			
#Put all the data for the hour into the 'new' data frame.
	for i in hourly_variables:
		n=0
		for item in theJSON["hourly"]["data"]:
			try:
				new.loc[0,str(n)+"."+i+".hourly"] =item[i]
				n+=1
			except:
				new.loc[0,str(n)+"."+i+".hourly"] =np.nan
				n+=1
	return new 