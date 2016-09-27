from datetime import date, timedelta as td
import datetime
import daily_weather as dw
import pandas as pd

d1 = date(2016,9,13)
d2 = date(2016,1,1)

delta= d1-d2
d = pd.DataFrame()
apikey= 'abc7928ccbb6a65eeba40bc7c10e1529'
for i in range(delta.days+1) :
	day =d2+td(days=i)
	da =day.isoformat()+'T00:00:00'	
	data = dw.daily_weather(lat= '41.878311', long='-87.616342',date=da,apikey=apikey)
	d.loc[i,'Date'] = day.isoformat()
	for j in data :
		d.loc[i,j] = data.loc[0,j]
	print(day.isoformat(),'\n')
d.to_csv('2016_weather.csv',index=False)