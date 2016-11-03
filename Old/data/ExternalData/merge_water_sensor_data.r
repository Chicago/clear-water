
# -------------------------------------
# --------- WATER SENSOR DATA ---------
# -------------------------------------

# get sensor data from Chicago Open Data Portal 
water_sensors <- read.csv("https://data.cityofchicago.org/api/views/qmqz-2xku/rows.csv?accessType=DOWNLOAD", stringsAsFactors = F, header = T)
sensor_locations <- read.csv("https://data.cityofchicago.org/api/views/g3ip-u8rb/rows.csv?accessType=DOWNLOAD", stringsAsFactors = F, header = T)

# merge sensor data
water_sensors <- merge(water_sensors, sensor_locations, by.x="Beach.Name", by.y="Sensor.Name")

# drop unecessary columns
drop <- c("Sensor.Type","Location")
water_sensors <- water_sensors[,!(names(water_sensors) %in% drop)]

# normalize names
water_sensors$Beach.Name <- substr(water_sensors$Beach.Name,1,nchar(water_sensors$Beach.Name)-6)

# match beaches with water and weather stations
beach_mappings <- read.csv("data/ExternalData/Beach_Mappings.csv", stringsAsFactors=FALSE)
waternames <- setNames(beach_mappings$water, beach_mappings$beach) 
beach_readings$Water.Station <- waternames[beach_readings$Client.ID]

# parse dates for water and weather
water_sensors$Date <- substr(water_sensors$Measurement.Timestamp,1,10)
water_sensors$Date<- as.Date(water_sensors$Date, format="%m/%d/%Y")

# Find average, max, and min values for all water and weather metrics per day per senser location
df_min_water_temp <- aggregate(water_sensors$Water.Temperature, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=min)
df_max_water_temp <- aggregate(water_sensors$Water.Temperature, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=max)
df_avg_water_temp <- aggregate(water_sensors$Water.Temperature, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=mean)
water_temp <- join_all(list(df_min_water_temp, df_max_water_temp, df_avg_water_temp), by=c("Group.1", "Group.2"))
names(water_temp) <- c("Water.Station", "Full_date", "Water.Temp.Min", "Water.Temp.Max", "Water.Temp.Mean")

df_min_turbidity <- aggregate(water_sensors$Turbidity, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=min)
df_max_turbidity <- aggregate(water_sensors$Turbidity, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=max)
df_avg_turbidity <- aggregate(water_sensors$Turbidity, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=mean)
turbidity <- join_all(list(df_min_turbidity, df_max_turbidity, df_avg_turbidity), by=c("Group.1", "Group.2"))
names(turbidity) <- c("Water.Station", "Full_date", "Turbidity.Min", "Trubidity.Max", "Turbidity.Mean")

df_min_transducer_depth <- aggregate(water_sensors$Transducer.Depth, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=min)
df_max_transducer_depth <- aggregate(water_sensors$Transducer.Depth, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=max)
df_avg_transducer_depth <- aggregate(water_sensors$Transducer.Depth, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=mean)
transducer_depth <- join_all(list(df_min_transducer_depth, df_max_transducer_depth, df_avg_transducer_depth), by=c("Group.1", "Group.2"))
names(transducer_depth) <- c("Water.Station", "Full_date", "Transducer.Depth.Min", "Transducer.Depth.Max", "Transducer.Depth.Mean")

df_min_wave_height <- aggregate(water_sensors$Wave.Height, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=min)
df_max_wave_height <- aggregate(water_sensors$Wave.Height, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=max)
df_avg_wave_height <- aggregate(water_sensors$Wave.Height, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=mean)
wave_height <- join_all(list(df_min_wave_height, df_max_wave_height, df_avg_wave_height), by=c("Group.1", "Group.2"))
names(wave_height) <- c("Water.Station", "Full_date", "Wave.Height.Min", "Wave.Height.Max", "Wave.Height.Mean")

df_min_wave_period <- aggregate(water_sensors$Wave.Period, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=min)
df_max_wave_period <- aggregate(water_sensors$Wave.Period, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=max)
df_avg_wave_period <- aggregate(water_sensors$Wave.Period, by=list(water_sensors$Beach.Name, water_sensors$Date), FUN=mean)
wave_period <- join_all(list(df_min_wave_period, df_max_wave_period, df_avg_wave_period), by=c("Group.1", "Group.2"))
names(wave_period) <- c("Water.Station", "Full_date", "Wave.Period.Min", "Wave.Period.Max", "Wave.Period.Mean")

water_sensors_daily <- join_all(list(water_temp, turbidity, transducer_depth, wave_height, wave_period), by=c("Water.Station", "Full_date"))

# merge with beach_readings
beach_readings <- merge(beach_readings, water_sensors_daily, all.x = TRUE)

# clean up workspace
rm(list = ls(pattern = "^df_min_"))
rm(list = ls(pattern = "^df_max_"))
rm(list = ls(pattern = "^df_avg_"))
rm(list=c("water_temp", "wave_period", "wave_height", "turbidity", "transducer_depth"))

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
  "Wave.Period.Min","Wave.Period.Max","Wave.Period.Mean"
)]
