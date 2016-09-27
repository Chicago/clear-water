def daily_weather(lat,long,date,apikey) :
	import requests
	import json
	import pandas as pd
	import numpy as np
	url = ('https://api.forecast.io/forecast/'+apikey+'/'+lat + ','+long +','+date)
	# url = ('https://api.forecast.io/forecast/'+'abc7928ccbb6a65eeba40bc7c10e1529'+'/'+'41.878311' + ','+'-87.616342' +','+'2016-05-03T00:00:00')
	resp = requests.get(url)
	theJSON = json.loads(resp.text)

	daily_variables = []
	hourly_variables = []
	new = pd.DataFrame()

	for key in theJSON["daily"]["data"][0]:
		daily_variables.append(key)

	for name in daily_variables:
		new[name]=np.nan
		
	for i in daily_variables:
		for item in theJSON["daily"]["data"]:
			new.loc[0,i] = item[i]

	for key in theJSON["hourly"]["data"][0]:
		hourly_variables.append(key)

	# for i in hourly_variables:
		# if i =='precipType' :
			# hourly_variables.remove('precipType')
		# elif i =='visibility' :
			# hourly_variables.remove('visibility')

	for name in hourly_variables:
		for j in range(24):
			new[str(j)+"."+name+".hourly"]=np.nan


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