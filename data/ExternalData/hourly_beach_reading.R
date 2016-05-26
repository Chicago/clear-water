require(httr)
require(dplyr)

top_of_hour <- function(posix.time){ # Calculates the top-of-hour to request data (in-case there is a lag)
  day <- format(posix.time, "%Y-%m-%d")
  hour <- format(posix.time, "%H")
  time <- strptime(paste0(day," ",hour,":00:00"), format = "%Y-%m-%d %H:%M:%S")
  epoc_time <- as.numeric(time)
  return(epoc_time)
}

hourly_beach_reading <- function(list_of_beaches, previous_beach_readings, API_key, time){ # Requests data from Forecast.io
  beaches <- read.csv(list_of_beaches, stringsAsFactors = FALSE)
  beachReadings <- read.csv(previous_beach_readings, stringsAsFactors = FALSE)
  for(j in 1:dim(beachReadings)[2]){
    beachReadings[ ,j] <- as.character(beachReadings[ ,j])
  }
  for(i in 1:dim(beaches)[1]){
    beach <- beaches[i, ]
    thisBeach <- GET(paste0("https://api.forecast.io/forecast/",API_key,"/",beaches[i,2],",",beaches[i,3],",",hourly_check_time))
    thisBeachFrame <- as.data.frame(content(thisBeach), stringsAsFactors = FALSE)
    thisBeachFrame$beach <- beach[i,1]
    for(j in 1:dim(thisBeachFrame)[2]){
      thisBeachFrame[ ,j] <- as.character(thisBeachFrame[ ,j])
    }
    beachReadings <- bind_rows(beachReadings, thisBeachFrame)
    i <- i + 1
  }
write.csv(beachReadings, file=previous_beach_readings, row.names = FALSE)  
}





