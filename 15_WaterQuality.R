#This will read in all the water sensor data that the city has on their website
#And change it into a data frame that we can use to join with the main data frame.
#To begin with the parks department had 6 Sensorsors working and in 2016 they 
#Cut the amount of sensors down to just 3

#Read in the data from Chicago City Website
water_quality_df <- read.socrata("https://data.cityofchicago.org/resource/qmqz-2xku.csv")

#Keep the variables that I want
water_quality_df<- data.frame("Date" = water_quality_df$Measurement.Timestamp,
                              "Client.ID" = water_quality_df$Beach.Name,
                              "Water.Temp"= water_quality_df$Water.Temperature,
                              "Turbidity"= water_quality_df$Turbidity,
                              "Wave.Height" = water_quality_df$Wave.Height,
                              "Wave.Period" = water_quality_df$Wave.Period)

#Change the date from a double to character to work with in DateTime formatting
water_quality_df$Date <- as.character(water_quality_df$Date)

#If we don't have a date with the reading we can't use it
water_quality_df <- water_quality_df[!(is.na(water_quality_df$Date)| water_quality_df$Date==""),]

#Clean the beachnames to match with the rest of the project
water_quality_df$Client.ID<-BeachNames(water_quality_df$Client.ID)

#Strip down the "Date" variable to pull out the Year, Day, and Month later
water_quality_df$Date<-strptime(water_quality_df$Date,
                            format="%Y-%m-%d %H:%M:%S")

#Pull out the "Year", "Day", and "Month" variables
water_quality_df$Year<-strftime(water_quality_df$Date,"%Y")
water_quality_df$Month<-strftime(water_quality_df$Date,format="%m")
water_quality_df$Day<-strftime(water_quality_df$Date,format="%d")
water_quality_df$Hour<-strftime(water_quality_df$Date,format="%H")

#There are quite a few duplicate rows, so we take them out
water_quality_df<-water_quality_df[!duplicated(water_quality_df),]

#The gather() function does not work unless "Date" is in POSIXct form
water_quality_df$Date <-as.POSIXct(water_quality_df$Date)

#Gather in all the varibles for each day so that we can rename the Variables
water_quality_df<-gather(water_quality_df,"code","value",c(3:6))

#Rename the variables in "Value_hourly_#hour" format
water_quality_df$code<- paste(water_quality_df$code,water_quality_df$Hour,sep="_hourly_")

#Drop the "Date and Hour Variables"
drops<-c("Date","Hour")
water_quality_df<-water_quality_df[,!(names(water_quality_df) %in% drops)]
rm(drops)

#Make an observation and not each hour
water_quality_df<-spread(water_quality_df,code,value)
