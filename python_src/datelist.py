#This program creates a csv '2016_weather' for a specific location which goes from Jan. 1 to Sept. 13. A DarkSky.net api key is needed to run this program.

from datetime import date, timedelta as td
import datetime
import daily_weather as dw
import pandas as pd


 #A DarkSky.net api key is needed to be input.
status = False
while status == False:
	apikey= str(input('Enter your Dark Sky APIkey:'))
	url = 'https://api.forecast.io/forecast/'+apikey+'/41.878311,-87.616342,'+datetime.date.today().isoformat()+'T00:00:00'
	r = requests.get(url)
	status = r.status_code == requests.codes.ok
	
d1 = date(2016,9,13) #End date
d2 = date(2016,1,1) #Start date 

delta= d1-d2 #Determine how many days are between the end date and start date
d = pd.DataFrame() #Create a data fame d

#This function goes into each day in the date range, uses daily_weather.py to grab the information that is needed to run the analyses in the future, and places it in the 'd' data frame.
for i in range(delta.days+1) :
	day =d2+td(days=i) #Getting the specific day
	da =day.isoformat()+'T00:00:00'	#create the datetime format needed for the darksky.net api.
	data = dw.daily_weather(lat= '41.878311', long='-87.616342',date=da,apikey=apikey) #run the daily_weather function that returns the weather for a specific day.
	d.loc[i,'Date'] = day.isoformat() #Add the date to the d data frame.
	for j in data :
		d.loc[i,j] = data.loc[0,j] #Put all the weather information from dark sky onto the d dataframe
		
d.to_csv('2016_weather.csv',index=False) #Put data frame d into 2016_weather.csv