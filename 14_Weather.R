print("Importing Weather Data")
#Load in the Weatherdata
weather_data <- read.csv("weather_data.csv",
                         stringsAsFactors=FALSE)


#Strip down the "date" variable to pull out the Year, Day, and Month later
weather_data$date<-strptime(weather_data$date,
                                       format="%Y-%m-%d")
#Pull out the "Year", "Day", and "Month" variables
weather_data$Year<-strftime(weather_data$date,"%Y")
weather_data$Month<-strftime(weather_data$date,format="%m")
weather_data$Day<-strftime(weather_data$date,format="%d")

#Remove the "date" Variable
drops<-c("date")
weather_data<-weather_data[,!(names(weather_data) %in% drops)]
rm(drops)
