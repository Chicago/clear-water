
print("Importing Water Level Data")
#Read in the Beach_Water_Levels.csv
Beach_Water_Levels <- read.csv("CSVs/Beach_Water_Levels.csv", stringsAsFactors=FALSE, as.is=TRUE)

#Strip down the "Date.Time" variable to pull out the Year, Day, and Month later
Beach_Water_Levels$Date.Time<-strptime(Beach_Water_Levels$Date.Time,
                                       format="%m/%d/%Y %H:%M")
#Pull out the "Year", "Day", and "Month" variables
Beach_Water_Levels$Year<-strftime(Beach_Water_Levels$Date.Time,"%Y")
Beach_Water_Levels$Month<-strftime(Beach_Water_Levels$Date.Time,format="%m")
Beach_Water_Levels$Day<-strftime(Beach_Water_Levels$Date.Time,format="%d")

#Remove the "Date.Time" Variable
drops<-c("Date.Time")
Beach_Water_Levels<-Beach_Water_Levels[,!(names(Beach_Water_Levels) %in% drops)]
rm(drops)
