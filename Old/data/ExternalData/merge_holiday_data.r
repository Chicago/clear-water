
# -------------------------------
# --------- HOLDIAY DATA --------
# -------------------------------

# what holidays are important for beaches?
# Memorial Day, Labor Day, Independence Day, Cinco de Mayo, Father's Day, Summer Solstice
holidays <- read.csv("data/ExternalData/Holidays.csv", stringsAsFactors = T, header=T)

holidays$Date <- as.Date(holidays$Date, format="%m-%d-%Y")

# merge with beach_readings
beach_readings <- merge(beach_readings, holidays, by.x="Full_date", by.y="Date", all.x = TRUE)

# create "holiday or not holiday" indicator variable for each record
beach_readings$Holiday.Flag <- ifelse(!is.na(beach_readings$Holiday),1,0)

# create "days since last holiday" variable 
## QUESTION - is this necessary or will a time series model be able to know this already?

sorted <- beach_readings[order(beach_readings$Client.ID, beach_readings$Full_date),] 

count <- 1
first_event_flag <- 0
current_beach <- sorted$Client.ID[1]
for (i in 1:nrow(sorted)) {
  if (!identical(sorted$Client.ID[i],current_beach)) {
    count <- 1
    first_event_flag <- 0
  }
  if (first_event_flag == 0 && sorted$Holiday.Flag[i] == 0) {
    sorted$Days.Since.Last.Holiday[i] <- NA
  }
  else if (first_event_flag == 0 && sorted$Holiday.Flag[i] == 1) {
    first_event_flag <- 1
  }
  if (first_event_flag == 1) {
    if (sorted$Holiday.Flag[i] == 1) {
      sorted$Days.Since.Last.Holiday[i] <- 0
      count <- 1
    }
    else {
      sorted$Days.Since.Last.Holiday[i] <- count
      count <- count + 1
    }
  }
  current_beach <- sorted$Client.ID[i]
}

beach_readings <- sorted

# clean up workspace
rm(list=c("sorted"))

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
  "Solar.Radiation.Min","Solar.Radiation.Max","Solar.Radiation.Mean",
  "Holiday","Holiday.Flag","Days.Since.Last.Holiday"
)] 


