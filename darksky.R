####--------------------------------------------------------------------------------------------------
####    DOWNLOAD WEATHER FROM DARK SKY
####--------------------------------------------------------------------------------------------------

library(jsonlite)
library(plyr)

#-----------------------------------------------------------------------------------------------------
# Build the basics of the URL for the Dark Sky API
#-----------------------------------------------------------------------------------------------------

# Enter your darksky secret key
key <- "your_key_here"

lhs <- "https://api.darksky.net/forecast/"
rhs <- "exclude=currently,flags"

#-----------------------------------------------------------------------------------------------------
# Obtain the location coordinates for the beaches
#-----------------------------------------------------------------------------------------------------

beaches <- read.csv("CSVs/cleanbeachnames.csv", stringsAsFactors = FALSE)
beaches <- beaches[,c("Short_Names",
                      "Latitude",
                      "Longitude")]
beaches <- beaches[complete.cases(beaches),]
beaches <- beaches[!duplicated(beaches),]

#-----------------------------------------------------------------------------------------------------
# Make API calls and build dataframe
#-----------------------------------------------------------------------------------------------------

weather_data <- data.frame()

date <- as.Date("2016/05/01")
end_date <- as.Date("2016/09/30")

## Maximum 1000 free API requests allowed per day
counter <- 1

## no need to change the next three (fields required by API)
hour <- "12"  # we are downloading by day, so this does not matter
minute <- "00" # we are downloading by day, so this does not matter
second <- "00" # we are downloading by day, so this does not matter

while (date <= end_date & counter <= 1000) {
  year <- format(date, "%Y")
  month <- format(date, "%m")
  day <- format(date, "%d")
  
  for (beach in beaches$Short_Names) {
    
    lat <- beaches$Latitude[beaches$Short_Names == beach]
    long <- beaches$Longitude[beaches$Short_Names == beach]
    
    darksky_url <- paste0(lhs,
                          key, "/",
                          lat, ",",
                          long, ",",
                          year, "-",
                          month, "-",
                          day, "T",
                          hour, ":",
                          minute, ":",
                          second, "?",
                          rhs)

    darksky_response <- fromJSON(darksky_url)
    counter <- counter + 1
    temp_df <- cbind(beach, darksky_response$daily$data)
    
    hourly_weather <- darksky_response$hourly$data
    
    for (row in c(1:nrow(hourly_weather))) {
      dat <- hourly_weather[row,]
      names(dat) <- paste(names(dat),"hourly",row,sep="_")
      temp_df <- cbind(temp_df, dat)
    }
    
    weather_data <- rbind.fill(weather_data,temp_df)
    
    ## doublecheck lats/longs before downloading
    ## test data integrity before using this data
    
  }
  date <- date + 1
}

#-----------------------------------------------------------------------------------------------------
# Export to CSV
#-----------------------------------------------------------------------------------------------------

weather_data$date <- as.POSIXct(weather_data$time, origin="1970-01-01")
write.table(weather_data,"weather_data.csv",sep=",")
