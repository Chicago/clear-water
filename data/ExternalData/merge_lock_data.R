
# read in lock data
lock_data <- read.csv("data/MWRD/lock_openings.csv",stringsAsFactors = FALSE)
lock_data <- na.omit(lock_data)
lock_data$begin_date <- as.Date(lock_data$begin_date,format="%m/%d/%Y")
lock_data$end_date <- as.Date(lock_data$end_date,format="%m/%d/%Y")

# create lock indicators for each region
# NOTE: the Obrien lock is never open during the days in our data so it is left out
lock_data$CRCW.Lock.Open <- ifelse(lock_data$CRCW>0,1,0)
lock_data$Wilmette.Lock.Open <- ifelse(lock_data$Wilmette>0,1,0)

# create variable for volume per day each lock was open
lock_data$CRCW.Lock.Volume <- lock_data$CRCW / lock_data$days_lock_open
lock_data$Wilmette.Lock.Volume <- lock_data$Wilmette / lock_data$days_lock_open

# create lock indicators for each of the 3 regions which have locks based on matching dates
library(sqldf)
beach_readings <- sqldf("SELECT a.*, b.'CRCW.Lock.Open', b.'Wilmette.Lock.Open', b.'CRCW.Lock.Volume', b.'Wilmette.Lock.Volume'
               FROM beach_readings as a LEFT JOIN lock_data as b
               ON a.Full_Date BETWEEN b.begin_date AND b.end_date")

# set NA values to 0
beach_readings$CRCW.Lock.Open[is.na(beach_readings$CRCW.Lock.Open)] <- 0
beach_readings$Wilmette.Lock.Open[is.na(beach_readings$Wilmette.Lock.Open)] <- 0
beach_readings$CRCW.Lock.Volume[is.na(beach_readings$CRCW.Lock.Volume)] <- 0
beach_readings$Wilmette.Lock.Volume[is.na(beach_readings$Wilmette.Lock.Volume)] <- 0

