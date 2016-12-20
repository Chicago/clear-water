
# -------------------------------------
# -------- WEATHER SENSOR DATA --------
# -------------------------------------

# get sensor data from Chicago Open Data Portal 
weather_sensors <- read.csv("https://data.cityofchicago.org/api/views/k7hf-8y75/rows.csv?accessType=DOWNLOAD", stringsAsFactors = F, header = T)
sensor_locations <- read.csv("https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD", stringsAsFactors = F, header = T)

# merge sensor data
weather_sensors <- merge(weather_sensors, sensor_locations, by.x="Station.Name", by.y="Sensor.Name")

# drop unecessary columns
drop <- c("Sensor.Type","Location")
weather_sensors <- weather_sensors[,!(names(weather_sensors) %in% drop)]

# normalize names
weather_sensors$Station.Name <- substr(weather_sensors$Station.Name,1,nchar(weather_sensors$Station.Name)-16)

# match beaches with water and weather stations
beach_mappings <- read.csv("data/ExternalData/Beach_Mappings.csv", stringsAsFactors=FALSE)
weathernames <- setNames(beach_mappings$weather, beach_mappings$beach) 
beach_readings$Weather.Station <- weathernames[beach_readings$Client.ID]

# parse dates for water and weather
weather_sensors$Date <- substr(weather_sensors$Measurement.Timestamp,1,10)
weather_sensors$Date<- as.Date(weather_sensors$Date, format="%m/%d/%Y")

# Find average, max, and min values for all water and weather metrics per day per senser location
df_min_air_temp <- aggregate(weather_sensors$Air.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_air_temp <- aggregate(weather_sensors$Air.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_air_temp <- aggregate(weather_sensors$Air.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
air_temp <- join_all(list(df_min_air_temp, df_max_air_temp, df_avg_air_temp), by=c("Group.1", "Group.2"))
names(air_temp) <- c("Weather.Station", "Full_date", "Air.Temp.Min", "Air.Temp.Max", "Air.Temp.Mean")

df_min_bulb_temp <- aggregate(weather_sensors$Wet.Bulb.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_bulb_temp <- aggregate(weather_sensors$Wet.Bulb.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_bulb_temp <- aggregate(weather_sensors$Wet.Bulb.Temperature, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
bulb_temp <- join_all(list(df_min_bulb_temp, df_max_bulb_temp, df_avg_bulb_temp), by=c("Group.1", "Group.2"))
names(bulb_temp) <- c("Weather.Station", "Full_date", "Wet.Bulb.Temp.Min", "Wet.Bulb.Temp.Max", "Wet.Bulb.Temp.Mean")

df_min_humidity <- aggregate(weather_sensors$Humidity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_humidity <- aggregate(weather_sensors$Humidity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_humidity <- aggregate(weather_sensors$Humidity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
humidity <- join_all(list(df_min_humidity, df_max_humidity, df_avg_humidity), by=c("Group.1", "Group.2"))
names(humidity) <- c("Weather.Station", "Full_date", "Humidity.Min", "Humidity.Max", "Humidity.Mean")

df_min_rain_intensity <- aggregate(weather_sensors$Rain.Intensity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_rain_intensity <- aggregate(weather_sensors$Rain.Intensity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_rain_intensity <- aggregate(weather_sensors$Rain.Intensity, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
rain_intensity <- join_all(list(df_min_rain_intensity, df_max_rain_intensity, df_avg_rain_intensity), by=c("Group.1", "Group.2"))
names(rain_intensity) <- c("Weather.Station", "Full_date", "Rain.Intensity.Min", "Rain.Intensity.Max", "Rain.Intensity.Mean")

df_min_interval_rain <- aggregate(weather_sensors$Interval.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_interval_rain <- aggregate(weather_sensors$Interval.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_interval_rain <- aggregate(weather_sensors$Interval.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
interval_rain <- join_all(list(df_min_interval_rain, df_max_interval_rain, df_avg_interval_rain), by=c("Group.1", "Group.2"))
names(interval_rain) <- c("Weather.Station", "Full_date", "Interval.Rain.Min", "Interval.Rain.Max", "Interval.Rain.Mean")

df_min_total_rain <- aggregate(weather_sensors$Total.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_total_rain <- aggregate(weather_sensors$Total.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_total_rain <- aggregate(weather_sensors$Total.Rain, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
total_rain <- join_all(list(df_min_total_rain, df_max_total_rain, df_avg_total_rain), by=c("Group.1", "Group.2"))
names(total_rain) <- c("Weather.Station", "Full_date", "Total.Rain.Min", "Total.Rain.Max", "Total.Rain.Mean")

df_min_wind_direction <- aggregate(weather_sensors$Wind.Direction, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_wind_direction <- aggregate(weather_sensors$Wind.Direction, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_wind_direction <- aggregate(weather_sensors$Wind.Direction, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
wind_direction <- join_all(list(df_min_wind_direction, df_max_wind_direction, df_avg_wind_direction), by=c("Group.1", "Group.2"))
names(wind_direction) <- c("Weather.Station", "Full_date", "Wind.Direction.Min", "Wind.Direction.Max", "Wind.Direction.Mean")

df_min_wind_speed <- aggregate(weather_sensors$Wind.Speed, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_wind_speed <- aggregate(weather_sensors$Wind.Speed, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_wind_speed <- aggregate(weather_sensors$Wind.Speed, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
wind_speed <- join_all(list(df_min_wind_speed, df_max_wind_speed, df_avg_wind_speed), by=c("Group.1", "Group.2"))
names(wind_speed) <- c("Weather.Station", "Full_date", "Wind.Speed.Min", "Wind.Speed.Max", "Wind.Speed.Mean")

df_min_barometric_pressure <- aggregate(weather_sensors$Barometric.Pressure, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_barometric_pressure <- aggregate(weather_sensors$Barometric.Pressure, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_barometric_pressure <- aggregate(weather_sensors$Barometric.Pressure, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
barometric_pressure <- join_all(list(df_min_barometric_pressure, df_max_barometric_pressure, df_avg_barometric_pressure), by=c("Group.1", "Group.2"))
names(barometric_pressure) <- c("Weather.Station", "Full_date", "Barometric.Pressure.Min", "Barometric.Pressure.Max", "Barometric.Pressure.Mean")

df_min_solar_radiation <- aggregate(weather_sensors$Solar.Radiation, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=min)
df_max_solar_radiation <- aggregate(weather_sensors$Solar.Radiation, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=max)
df_avg_solar_radiation <- aggregate(weather_sensors$Solar.Radiation, by=list(weather_sensors$Station.Name, weather_sensors$Date), FUN=mean)
solar_radiation <- join_all(list(df_min_solar_radiation, df_max_solar_radiation, df_avg_solar_radiation), by=c("Group.1", "Group.2"))
names(solar_radiation) <- c("Weather.Station", "Full_date", "Solar.Radiation.Min", "Solar.Radiation.Max", "Solar.Radiation.Mean")

weather_sensors_daily <- join_all(list(air_temp, bulb_temp, humidity, rain_intensity, interval_rain, total_rain, wind_direction, wind_speed, barometric_pressure, solar_radiation), by=c("Weather.Station", "Full_date"))

# merge with beach_readings
beach_readings <- merge(beach_readings, weather_sensors_daily, all.x = TRUE)

# clean up workspace
rm(list = ls(pattern = "^df_min_"))
rm(list = ls(pattern = "^df_max_"))
rm(list = ls(pattern = "^df_avg_"))
rm(list=c("air_temp", "bulb_temp", "humidity", "rain_intensity", "interval_rain", "total_rain", "wind_direction", "wind_speed", "barometric_pressure", "solar_radiation"))

# clean up order
beach_readings <- beach_readings[order(-as.numeric(beach_readings$Full_date)), c(
  "Client.ID",  "Full_date", "Year", "Date",
  "Laboratory.ID", "Reading.1", "Reading.2",
  "Escherichia.coli", "Units", "Sample.Collection.Time",
  "Weekday", "Month", "Day",
  "Drek_Reading", "Drek_Prediction", "Drek_Worst_Swim_Status",
  "e_coli_geomean_actual_calculated", "elevated_levels_actual_calculated", "Drek_elevated_levels_predicted_calculated",
  "Water.Station", "Water.Temp.Min", "Water.Temp.Max","Water.Temp.Mean",             
  "Turbidity.Min","Trubidity.Max", "Turbidity.Mean",                          
  "Transducer.Depth.Min","Transducer.Depth.Max", "Transducer.Depth.Mean",
  "Wave.Height.Min","Wave.Height.Max", "Wave.Height.Mean",
  "Wave.Period.Min","Wave.Period.Max","Wave.Period.Mean",
  "Weather.Station", "Air.Temp.Min","Air.Temp.Max","Air.Temp.Mean",
  "Wet.Bulb.Temp.Min","Wet.Bulb.Temp.Max","Wet.Bulb.Temp.Mean",
  "Humidity.Min","Humidity.Max","Humidity.Mean",
  "Rain.Intensity.Min","Rain.Intensity.Max","Rain.Intensity.Mean",
  "Interval.Rain.Min","Interval.Rain.Max","Interval.Rain.Mean",
  "Total.Rain.Min","Total.Rain.Max","Total.Rain.Mean",
  "Wind.Direction.Min","Wind.Direction.Max","Wind.Direction.Mean",
  "Wind.Speed.Min","Wind.Speed.Max","Wind.Speed.Mean",
  "Barometric.Pressure.Min","Barometric.Pressure.Max","Barometric.Pressure.Mean",
  "Solar.Radiation.Min","Solar.Radiation.Max","Solar.Radiation.Mean"
)]

