'''
This program creates a .csv  for a specific location right off of Jackson.
The start date and end date are chosen by the user. A DarkSky.net api key is
needed to run this function.\

`start_date` and `end_date` are in the datetime.date() format
'''
def datelist(start_date, end_date) :
	from datetime import date, timedelta as td
	import datetime
	import daily_weather as dw
	import pandas as pd
	import requests

	d1 = end_date #End date
	d2 = start_date #Start date 


	#A DarkSky.net api key is needed to be input.

	status = False
	while status == False:
		apikey= str(input('Enter your Dark Sky APIkey:'))
		url = 'https://api.forecast.io/forecast/'+apikey+'/41.878311,-87.616342,'+datetime.date.today().isoformat()+'T00:00:00'
		r = requests.get(url)
		status = r.status_code == requests.codes.ok
		
	#Determine how many days are between the end date and start date
	delta= d1-d2
	d = pd.DataFrame() #Create a data fame d
	'''
	This function goes into each day in the date range, uses daily_weather.py
	to grab the information that is needed to run the analyses in the future,
	and places it in the 'd' data frame.
	'''
	for i in range(delta.days+1) :
		day =d2+td(days=i) #Getting the specific day
		
		#create the datetime format needed for the darksky.net api.
		da =day.isoformat()+'T00:00:00'	
		
		#run the daily_weather function that returns the 
		#weather for a specific day.
		data = dw.daily_weather(lat= '41.878311', long='-87.616342',date=da,apikey=apikey)
		
		d.loc[i,'Date'] = day.isoformat() #Add the date to the d data frame.
		
		#Put all the weather information from dark sky onto the d dataframe
		for j in data :
			d.loc[i,j] = data.loc[0,j]
			
	#Put data frame d into 2016_weather.csv		
	d.to_csv(d2.isoformat()+'_to_'+d1.isoformat()+'.csv',index=False)
	
	return d