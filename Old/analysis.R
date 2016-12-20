source("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/split_sheets.R")

library(lubridate)
library(RSocrata)

# Read-in data
df2006 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2006 Lab Results.xls", 2006)
df2007 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2007 Lab Results.xls", 2007)
df2008 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2008 Lab Results.xls", 2008)
df2009 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2009 Lab Results.xls", 2009)
df2010 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2010 Lab Results.xls", 2010)
df2011 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2011 Lab Results.xls", 2011)
df2012 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2012 Lab Results.xls", 2012)
df2013 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2013 Lab Results.xls", 2013)
df2014 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2014 Lab Results.xls", 2014)
df2015 <- split_sheets("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/2015 Lab Results.xlsx", 2015)
# The 2016 read-in will bring in all readings after 1/1/2016
df2016 <- read.socrata("https://data.cityofchicago.org/resource/2ivx-z93u.csv?$where=culture_sample_1_timestamp > '2015-12-31T23:59:59'")
df2016 <- data.frame("Year" = 2016, "Date" = df2016$Culture.Sample.1.Timestamp,
                     "Laboratory.ID" = df2016$Culture.Test.ID, "Client.ID" = df2016$Beach, 
                     "Reading.1" = df2016$Culture.Sample.1.Reading, "Reading.2" = df2016$Culture.Sample.2.Reading,
                     "Escherichia.coli" = df2016$Culture.Reading.Mean, "Units" = NA, 
                     "Sample.Collection.Time" = df2016$Culture.Sample.2.Timestamp)
df2016$Date <- as.character(df2016$Date, format='%B %d')

# Combine Data
beach_readings <- rbind(df2006, df2007, df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015, df2016)

# Clean Data
## Remove greater or less-than markings
beach_readings$Reading.1 <- factor(gsub(">", "", beach_readings$Reading.1)) # Remove greater-than marks
beach_readings$Reading.2 <- factor(gsub(">", "", beach_readings$Reading.2)) # Remove greater-than marks
beach_readings$Escherichia.coli <- factor(gsub(">", "", beach_readings$Escherichia.coli)) # Remove greater-than marks
beach_readings$Reading.1 <- factor(gsub("<", "", beach_readings$Reading.1)) # Remove less-than marks
beach_readings$Reading.2 <- factor(gsub("<", "", beach_readings$Reading.2)) # Remove less-than marks
beach_readings$Escherichia.coli <- factor(gsub("<", "", beach_readings$Escherichia.coli)) # Remove less-than marks

beach_readings <- unite_(beach_readings, "Full_date", c("Date", "Year"), sep=" ", remove=F)
beach_readings$Full_date <- as.Date(beach_readings$Full_date, format="%B %d %Y") #now dates are sortable
beach_readings$Weekday <- weekdays(beach_readings$Full_date) #add day of week
beach_readings$Month <- format(beach_readings$Full_date,"%B")
beach_readings$Day <- format(beach_readings$Full_date, "%d")

##Remove problematic dates
beach_readings <- beach_readings[-which(beach_readings$Full_date %in% c(as.Date("2006-07-06"), as.Date("2006-07-08"), as.Date("2006-07-09"))),]

##Remove 6 duplicates (12 total observations) -- beaches that have more than one reading on a day
beach_readings=beach_readings[-which(beach_readings$Full_date=="2006-07-19" & beach_readings$Client.ID=="Jarvis/ Fargo"),]
beach_readings=beach_readings[-which(beach_readings$Full_date=="2006-08-24" & beach_readings$Client.ID=="Jarvis"),]
beach_readings=beach_readings[-which(beach_readings$Full_date=="2006-07-19" & beach_readings$Client.ID=="Hollywood/Ostermann"),]
beach_readings=beach_readings[-which(beach_readings$Full_date=="2007-06-08" & beach_readings$Client.ID=="Hollywood/Thorndale*"),]
beach_readings=beach_readings[-which(beach_readings$Full_date=="2008-06-08" & beach_readings$Client.ID=="Hollywood/Thorndale*"),]
beach_readings=beach_readings[-which(beach_readings$Full_date=="2006-08-08" & beach_readings$Client.ID=="Loyola"),]

##Remove 66 observations with date specified as "PM" -- which creates more than one reading per beach on a day
beach_readings=beach_readings[!grepl("PM",beach_readings$Date),]

##Normalize beach names using names found on cpdbeaches.com
cleanbeachnames <- read.csv("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/cleanbeachnames.csv", stringsAsFactors=FALSE)
changenames <- setNames(cleanbeachnames$New, cleanbeachnames$Old) 
beach_readings$Client.ID <- sapply(beach_readings$Client.ID, function (x) gsub("^\\s+|\\s+$", "", x)) #delete leading and trailing spaces
beach_readings$Client.ID <- changenames[beach_readings$Client.ID]  

##Clean Drek Data so they match beach_readings$Client.ID
drekdata <- read.csv("data/DrekBeach/daily_summaries_drekb.csv", stringsAsFactors = F)
names(drekdata) <- c("Beach", "Date", "Drek_Reading", "Drek_Prediction", "Drek_Worst_Swim_Status")
drekdata$Date <- as.Date(drekdata$Date, "%m/%d/%Y")
drekdata$Beach <- sapply(drekdata$Beach, function (x) gsub("^\\s+|\\s+$", "", x))
drekdata$Beach <- changenames[drekdata$Beach] 

##add 2016 USGS/EPA predictions from data portal
predictions_2016 <- read.socrata("https://data.cityofchicago.org/resource/t62e-8nvc.csv")

#map values so we have predictions for all beaches, per CPD standards
predictions_12th <- data.frame("Beach" = "12th",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "Ohio"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Ohio"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Ohio"])
predictions_31st <- data.frame("Beach" = "31st",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "63rdStreet"])
predictions_39th <- data.frame("Beach" = "39th",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "63rdStreet"])
predictions_57th <- data.frame("Beach" = "57th",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "63rdStreet"])
predictions_39th <- data.frame("Beach" = "39th",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "63rdStreet"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "63rdStreet"])
predictions_Albion <- data.frame("Beach" = "Albion",
                               "Date" = predictions_2016$Date[predictions_2016$Beach == "Leone"],
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Leone"],
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Leone"])
predictions_Howard <- data.frame("Beach" = "Howard",
                                 "Date" = predictions_2016$Date[predictions_2016$Beach == "Leone"],
                                 "Drek_Reading" = NA,
                                 "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Leone"],
                                 "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Leone"])
predictions_Jarvis <- data.frame("Beach" = "Jarvis",
                                 "Date" = predictions_2016$Date[predictions_2016$Beach == "Leone"],
                                 "Drek_Reading" = NA,
                                 "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Leone"],
                                 "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Leone"])
predictions_Juneway <- data.frame("Beach" = "Juneway",
                                 "Date" = predictions_2016$Date[predictions_2016$Beach == "Leone"],
                                 "Drek_Reading" = NA,
                                 "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Leone"],
                                 "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Leone"])
predictions_North <- data.frame("Beach" = "North",
                                 "Date" = predictions_2016$Date[predictions_2016$Beach == "OakStreet"],
                                 "Drek_Reading" = NA,
                                 "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "OakStreet"],
                                 "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "OakStreet"])
predictions_Rogers <- data.frame("Beach" = "Rogers",
                                "Date" = predictions_2016$Date[predictions_2016$Beach == "Leone"],
                                "Drek_Reading" = NA,
                                "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "Leone"],
                                "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "Leone"])
predictions_SouthShore <- data.frame("Beach" = "South Shore",
                                 "Date" = predictions_2016$Date[predictions_2016$Beach == "63rdStreet"],
                                 "Drek_Reading" = NA,
                                 "Drek_Prediction" = predictions_2016$Predicted.Level[predictions_2016$Beach == "63rdStreet"],
                                 "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory[predictions_2016$Beach == "63rdStreet"])

#align column names for merging
predictions_2016 <- data.frame("Beach" = predictions_2016$Beach.Name,
                               "Date" = predictions_2016$Date,
                               "Drek_Reading" = NA,
                               "Drek_Prediction" = predictions_2016$Predicted.Level,
                               "Drek_Worst_Swim_Status" = predictions_2016$Swim.Advisory)

predictions_2016 <- rbind(predictions_2016, predictions_12th, predictions_31st, predictions_39th,
                          predictions_57th, predictions_Albion, predictions_Howard, predictions_Jarvis, 
                          predictions_Juneway, predictions_North, predictions_Rogers, predictions_SouthShore) 

drekdata <- rbind(drekdata, predictions_2016)
##Merge drek with beach_readings 
beach_readings <- merge(beach_readings, drekdata, by.x = c("Client.ID", "Full_date"), by.y = c("Beach", "Date"), all.x=T)

##Final Clean of beach names for consistency from 2006 to 2015 -- now the the 20 beaches in the data set are the same as those with readings from 2010 through 2015
changenames.2 <- setNames(cleanbeachnames$Short_Names, cleanbeachnames$New) 
beach_readings$Beach_Name <- beach_readings$Client.ID
beach_readings$Beach_Name <- changenames.2[beach_readings$Beach_Name] 

#Remove all but instances with a beach name and at least 1 reading so that we only keep duplicate beaches WITH readings data (re: next line that removes duplicates)
beach_readings=beach_readings[which((!is.na(beach_readings$Reading.1) | !is.na(beach_readings$Reading.2)) & !is.na(beach_readings$Beach_Name)),]

#Remove all instances where same beach has more than one reading on a day, keeping only 1st instance - because some beaches merged during final name clean (e.g. hartigan, north shore, and tobey prinz become Albion)
beach_readings=beach_readings[!duplicated(beach_readings[c("Full_date", "Beach_Name")]), ] #keeps first instance, gets rid of dups

### Change readings to numeric data
beach_readings$Reading.1 <- as.numeric(as.character(beach_readings$Reading.1))
beach_readings$Reading.2 <- as.numeric(as.character(beach_readings$Reading.2))
beach_readings$Escherichia.coli <- as.numeric(as.character(beach_readings$Escherichia.coli))

###Remove outlier with 6488.0 value
beach_readings <- beach_readings[-which(beach_readings$Reading.2==6488.0),]

# Create measure variables
beach_readings$e_coli_geomean_actual_calculated <- round(apply(cbind(beach_readings$Reading.1,beach_readings$Reading.2), 1, geometric.mean, na.rm=T), 1) 

#create 1/0 for advisory at or over 235
beach_readings$elevated_levels_actual_calculated <- ifelse(beach_readings$e_coli_geomean_actual_calculated >= 235, 1, 0)
beach_readings$Drek_elevated_levels_predicted_calculated <- ifelse(beach_readings$Drek_Prediction >= 235, 1, 0)


# Bring in water sensor data (only available for last couple of years)
source("data/ExternalData/merge_water_sensor_data.r")

# Bring in weather sensor data (only available for last couple of years)
source("data/ExternalData/merge_weather_sensor_data.r")

# Bring in holiday data (only summer holidays)
source("data/ExternalData/merge_holiday_data.r")

# Bring in lock opening data
source("data/ExternalData/merge_lock_data.r")

#Beach_Name is lost when brining in water/weather/holiday/lock data -- this brings Beach_Name back in
beach_readings$Beach_Name <- beach_readings$Client.ID
beach_readings$Beach_Name <- changenames.2[beach_readings$Beach_Name]

#Bring in forecast.io daily weather data using cleaned Beach_Name
forecast_daily <- read.csv("data/ExternalData/forecastio_daily_weather.csv", stringsAsFactors = FALSE, row.names=NULL, header = T)
premerge_weather_2016 <- read.csv("python_src/2016_weather.csv", stringsAsFactors = FALSE, row.names=NULL, header = T)
beaches_2016 <- read.csv("data/ExternalData/Beach_Weather_2016_Mapping.csv", stringsAsFactors = FALSE, row.names=NULL, header = T)
beaches_2016$Date <- as.Date(beaches_2016$Date, "%m/%d/%Y")
beaches_2016$Date <- as.character(beaches_2016$Date)
merged_weather_2016 <- merge(beaches_2016, premerge_weather_2016)
forecast_daily_2016 <- data.frame("time" = merged_weather_2016$Date,
                                  "summary" = merged_weather_2016$summary,
                                  "icon" = merged_weather_2016$icon,
                                  "sunriseTime" = merged_weather_2016$sunriseTime,
                                  "sunsetTime" = merged_weather_2016$sunsetTime,
                                  "moonPhase" = merged_weather_2016$moonPhase,
                                  "precipIntensity" = merged_weather_2016$precipIntensity,
                                  "precipIntensityMax" = merged_weather_2016$precipIntensityMax,
                                  "precipProbability" = merged_weather_2016$precipProbability,
                                  "temperatureMin" = merged_weather_2016$temperatureMin,
                                  "temperatureMinTime" = merged_weather_2016$temperatureMinTime,
                                  "temperatureMax" = merged_weather_2016$temperatureMax,
                                  "temperatureMaxTime" = merged_weather_2016$temperatureMaxTime,
                                  "apparentTemperatureMin" = merged_weather_2016$apparentTemperatureMin,
                                  "apparentTemperatureMinTime" = merged_weather_2016$apparentTemperatureMinTime,
                                  "apparentTemperatureMax" = merged_weather_2016$apparentTemperatureMax,
                                  "apparentTemperatureMaxTime" = merged_weather_2016$apparentTemperatureMaxTime,
                                  "dewPoint" = merged_weather_2016$dewPoint,
                                  "humidity" = merged_weather_2016$humidity,
                                  "windSpeed" = merged_weather_2016$windSpeed,
                                  "windBearing" = merged_weather_2016$windBearing,
                                  "visibility" = merged_weather_2016$visibility,
                                  "cloudCover" = merged_weather_2016$cloudCover,
                                  "pressure" = merged_weather_2016$pressure,
                                  "beach" = merged_weather_2016$beach,
                                  "precipIntensityMaxTime" = merged_weather_2016$precipIntensityMaxTime,
                                  "precipType" = merged_weather_2016$precipType, stringsAsFactors=FALSE)
forecast_daily <- rbind (forecast_daily, forecast_daily_2016)
forecast_daily <- unique(forecast_daily)
forecast_daily$Beach_Name <- forecast_daily$beach
forecast_daily$Beach_Name <- changenames.2[forecast_daily$Beach_Name] #this cleans the beach names
forecast_daily=forecast_daily[!duplicated(forecast_daily[c("time", "Beach_Name")]), ] #remove duplicates, for example instances where North Shore & Hartigan appear on same day, due to merge on previous line
beach_readings <- merge(x=beach_readings, y=forecast_daily, by.x=c("Beach_Name", "Full_date"), by.y=c("Beach_Name", "time"), all.x = T, all.y = T)

#so that Client.ID is now the cleaned beaches, and we don't lose the old names
beach_readings$Old_Client.ID=beach_readings$Client.ID
beach_readings$Client.ID=beach_readings$Beach_Name
beach_readings=beach_readings[-which(colnames(beach_readings) %in% "Beach_Name")]

##Add time variables - most were lost during the weather/water/holiday/lock merge
beach_readings$Year <- as.numeric(format(beach_readings$Full_date, "%Y"))
beach_readings$Month <- format(beach_readings$Full_date,"%B")
beach_readings$Date <- format(beach_readings$Full_date,"%B %d")
beach_readings$Weekday <- weekdays(beach_readings$Full_date)  
beach_readings$Day_of_year <- as.numeric(format(beach_readings$Full_date, "%j"))  
beach_readings$Week<- format(beach_readings$Full_date, "%W")  
beach_readings$Day <- format(beach_readings$Full_date, "%d") #rename to Day_of_month? Other code may depend on this name
beach_readings$Weekday_code <- as.integer(wday(beach_readings$Full_date))
beach_readings$Month_code <- as.integer(as.character(beach_readings$Full_date, format="%m"))
beach_readings$Beach_code <- as.integer(sapply(beach_readings$Client.ID, function(x) {
                                      switch(x,
                                             "12th" = 1,
                                             "31st" = 2,
                                             "39th" = 3,
                                             "57th" = 4,
                                             "63rd" = 5,
                                             "Albion" = 6,
                                             "Calumet" = 7,
                                             "Foster" = 8,
                                             "Howard" = 9,
                                             "Jarvis" = 10,
                                             "Juneway" = 11,
                                             "Leone" = 12,
                                             "Montrose" = 13,
                                             "North Avenue" = 14,
                                             "Oak Street" = 15,
                                             "Ohio" = 16,
                                             "Osterman" = 17,
                                             "Rainbow" = 18,
                                             "Rogers" = 19,
                                             "South Shore" = 20)
                                      }
                                    ))


# Build naive logit model (today like yesterday)
# -----------------------------------------------------------

beach_readings_mod <- beach_readings[!is.na(beach_readings$Client.ID),]
beach_readings_mod <- beach_readings_mod[order(beach_readings_mod$Client.ID, beach_readings_mod$Full_date),]

# create "high reading" and "low reading"
beach_readings_mod$High.Reading <- mapply(max, beach_readings_mod$Reading.1, beach_readings_mod$Reading.2)
beach_readings_mod$Low.Reading <- mapply(min, beach_readings_mod$Reading.1, beach_readings_mod$Reading.2)

# create columns for yesterday's readings
library(useful)
temp <- split(beach_readings_mod, beach_readings_mod$Client.ID)
for (i in 1:length(temp)) {
  temp[[i]] <- shift.column(temp[[i]], columns=c("High.Reading","Low.Reading","e_coli_geomean_actual_calculated"), newNames=c("Yesterday.High.Reading", "Yesterday.Low.Reading", "Yesterday.Geomean"), len=1L, up=FALSE)
}
beach_readings_mod <- do.call("rbind", temp)

# use only records without NAs in predictors or response
beach_readings_mod <- beach_readings_mod[!is.na(beach_readings_mod$Yesterday.High.Reading) & !is.na(beach_readings_mod$Yesterday.Low.Reading) & !is.na(beach_readings_mod$Yesterday.Geomean)& !is.na(beach_readings_mod$elevated_levels_actual_calculated),]

# get train and test set
set.seed(12345)
smp_size <- floor(0.75 * nrow(beach_readings_mod))
train_ind <- sample(seq_len(nrow(beach_readings_mod)), size = smp_size)
train <- beach_readings_mod[train_ind, ]
test <- beach_readings_mod[-train_ind, ]

# fit naive logit model to training set
#fit <- glm(elevated_levels_actual_calculated ~ Yesterday.High.Reading + Yesterday.Low.Reading, data=train, family=binomial())
fit <- glm(elevated_levels_actual_calculated ~ Yesterday.Geomean, data=train, family=binomial())
summary(fit)

# evaluate model on test set
pred.prob=predict(fit, newdata=test, type="response")
pred.elevated <- rep(0,nrow(test))
pred.elevated[pred.prob>.5]=1
test$prediction <- pred.elevated
confmatrix=table(pred.elevated,test$elevated_levels_actual_calculated)
confmatrix
Recall=confmatrix[2,2]/(confmatrix[2,2]+confmatrix[1,2])
Recall # 2%
Precision=confmatrix[2,2]/(confmatrix[2,2]+confmatrix[2,1])
Precision # 32%
Fscore=(2*Precision*Recall)/(Precision+Recall)
Fscore # 0.04
Misclassification=1-(sum(diag(confmatrix))/nrow(test))
Misclassification # 14%

# -----------------------------------------------------------


# Calculate confusion matrix for 2015 (EPA model)

beach_readings_2015 <- beach_readings[beach_readings$Year==2015 & 
                                         !is.na(beach_readings$Reading.1) & 
                                         !is.na(beach_readings$Reading.2) &
                                         !is.na(beach_readings$Drek_Prediction)
                                       , ]

###@ Analyze the relationship between Reading.1 and Reading.2 in 2015
plot(beach_readings_2015$Reading.1, beach_readings_2015$Reading.2)

#### True positive -- correctly identifying elevated levels
actual_positive <- sum(beach_readings_2015$elevated_levels_actual_calculated)
true_positive <- beach_readings_2015[beach_readings_2015$Drek_elevated_levels_predicted_calculated == 1 &
                                       beach_readings_2015$elevated_levels_actual_calculated == 1, ]$Drek_elevated_levels_predicted_calculated
true_positive_perc <- sum(true_positive) / actual_positive

#### True negative -- correctly identifying non-elevated levels
actual_negative <- nrow(beach_readings_2015) - sum(actual_positive)
true_negative <- beach_readings_2015[beach_readings_2015$Drek_elevated_levels_predicted_calculated == 0 &
                                       beach_readings_2015$elevated_levels_actual_calculated == 0, ]$Drek_elevated_levels_predicted_calculated
true_negative_perc <- length(true_negative) / actual_negative

#### False negative -- failing to predict elevated levels
false_negative <- beach_readings_2015[beach_readings_2015$Drek_elevated_levels_predicted_calculated == 0 &
                                        beach_readings_2015$elevated_levels_actual_calculated == 1, ]$Drek_elevated_levels_predicted_calculated
false_negative_perc <- length(false_negative) / actual_negative # Exposing those to e coli

#### False positive -- incorrectly identifying elevated levels
false_positive <- beach_readings_2015[beach_readings_2015$Drek_elevated_levels_predicted_calculated == 1 &
                                        beach_readings_2015$elevated_levels_actual_calculated == 0, ]$Drek_elevated_levels_predicted_calculated
false_positive_perc <- length(false_positive) / actual_positive # Ruins the fun for the day

confusion_matrix <- table(true_positive_perc, true_negative_perc, false_positive_perc, false_negative_perc)

# Residual Analysis
resid <- beach_readings_2015$Drek_Prediction - beach_readings_2015$e_coli_geomean_actual_calculated
hist(resid, breaks = 30, main="Difference between predicted value and actual \n (negative denotes underestimate)")
summary(resid)

# Analytics functions
prCurve <- function(truth, predicted_values) {
    recalls = c()
    precisions = c()
    for (threshold in seq(0, 500, 20)) {
        recalls = c(recalls, recall(truth, predicted_values >= threshold))
        precisions = c(precisions, precision(truth, predicted_values >= threshold))
    }
    lines(recalls ~ precisions)
}


recall <- function(truth, predict) {
    return(sum(predict[truth])/sum(truth))
}

precision <- function(truth, predict) {
    return(sum(predict[truth])/sum(predict))
}

measures <- read.csv('data/DrekBeach/daily_summaries_drekb.csv')

measures$Date <- as.Date(measures$Date, '%m/%d/%Y')

measures$tomorrow <- measures$Date + 1

measures <- merge(measures, measures[, !(names(measures) %in% c("Date"))],
                  by.x=c('Beach', 'Date'),
                  by.y=c('Beach', 'tomorrow'))

measures <- measures[,c(1,2,3,4,5,8,9)]

names(measures) <- c("beach", "date", "reading", "prediction", "status",
                     "yesterday_reading", "yesterday_prediction")

true_advisory_days <- measures$reading > 235

plot(c(0,1), c(0,1), type="n")

prCurve(true_advisory_days,  measures$yesterday_reading)

prCurve(true_advisory_days,  measures$prediction)

model.naive <- glm(reading ~ yesterday_reading*beach + weekdays(date)*beach, measures, family='poisson')
summary(model.naive)

prCurve(true_advisory_days,  exp(predict(model.naive)))

model.forecast <- glm(reading ~ prediction*beach + weekdays(date)*beach + months(date), measures, family='poisson')
summary(model.forecast)

prCurve(true_advisory_days,  exp(predict(model.forecast)))

#This function can be called with or without the vector of column names, names_of_columns_to_shift
#without the argument names_of_columns_to_shift, it defaults to all numeric columns in original_data_frame
shift_previous_data <- function(number_of_observations, original_data_frame, names_of_columns_to_shift=FALSE) {
  merged_data_frame <- data.frame()
  #remove rows where Client.ID == NA
  clean_data_frame <- original_data_frame[!is.na(original_data_frame$Client.ID),]         
  #subset by year
  for (year in unique(clean_data_frame$Year)) {        
    readings_by_year <- clean_data_frame[clean_data_frame$Year == year,]
    #subset by beach within year
    for (beach in unique(readings_by_year$Client.ID) ) {     
      readings_by_beach <- readings_by_year[readings_by_year$Client.ID == beach,]
      readings_by_beach <- readings_by_beach[order(readings_by_beach$Full_date),]
      #if no columns provided, use default (all numeric columns)
      if (names_of_columns_to_shift[1] == FALSE) {        
        readings_by_beach_columns <- readings_by_beach[sapply(readings_by_beach, is.numeric)]
        names_of_columns_to_shift <- colnames(readings_by_beach_columns)
      }
      #build new column names
      for (column in names_of_columns_to_shift) {     
        new_column_name <- paste(number_of_observations,"daysPrior",column,sep=".")
        new_column_values <- vector()
        #build new columns
        #for first n rows, use NA bc no prior data to use
        for (n in 1:number_of_observations) {         
          new_column_values[n] <- NA            
        }
        #add previous data to new columns
        for (i in number_of_observations:nrow(readings_by_beach)) {  
          new_column_values <- c(new_column_values, readings_by_beach[,column][i-n])
        }
        #merge new columns with subsetted year/beach data
        readings_by_beach <- cbind(readings_by_beach, new_column_values)    
        colnames(readings_by_beach)[colnames(readings_by_beach)=="new_column_values"] <- new_column_name
      }
      #rebuild original dataframe adding the merged data from above
      merged_data_frame <- rbind(merged_data_frame, readings_by_beach)  
    }
  }
  merged_data_frame
}

#Principal Component Analysis

beach_readings_pca <- beach_readings
cols_to_remove <- c("Transducer.Depth.Min", "Transducer.Depth.Max", "Transducer.Depth.Mean", "Rain.Intensity.Min", "Interval.Rain.Min", "Holiday.Flag", "precipIntensityMaxTime")
beach_readings_pca <- beach_readings_pca[,!names(beach_readings_pca) %in% cols_to_remove]
beach_readings_pca_shifted <- shift_previous_data(1,beach_readings_pca)
new_shifted_col_names <- setdiff(colnames(beach_readings_pca_shifted),colnames(beach_readings_pca))
beach_readings_pca_shifted_new_only <- beach_readings_pca_shifted[,new_shifted_col_names]
cols_to_remove <- c("1.daysPrior.Year", "1.daysPrior.Reading.1", "1.daysPrior.Reading.2", "1.daysPrior.Escherichia.coli", "1.daysPrior.Drek_Reading", "1.daysPrior.Drek_Prediction", "1.daysPrior.e_coli_geomean_actual_calculated", "1.daysPrior.elevated_levels_actual_calculated", "1.daysPrior.Drek_elevated_levels_predicted_calculated")
beach_readings_pca_shifted_new_only <- beach_readings_pca_shifted_new_only[,!names(beach_readings_pca_shifted_new_only) %in% cols_to_remove]
beach_readings_pca <- cbind(beach_readings_pca_shifted$Reading.1, beach_readings_pca_shifted$Reading.2, beach_readings_pca_shifted$e_coli_geomean_actual_calculated, beach_readings_pca_shifted_new_only)
names(beach_readings_pca)[1:3] <- c("Reading.1", "Reading.2", "e_coli_geomean_actual_calculated")
beach_readings_pca <-  beach_readings_pca[,sapply(beach_readings_pca, is.numeric)]
na_count <- sapply(beach_readings_pca, function(y) sum(is.na(y))) #analyze NAs
beach_readings_pca <- beach_readings_pca[,na_count < 10000] #enforce NA maximum
beach_readings_pca <- beach_readings_pca[complete.cases(beach_readings_pca),] #remove NAs
beach_readings_pca <- beach_readings_pca[complete.cases(beach_readings_pca),]
beach_readings_pca <- scale(beach_readings_pca)
pca <- prcomp(beach_readings_pca)
plot(pca, type = "l")
aload <- abs(pca$rotation[,1:5])
relative_contribution_to_PC <- sweep(aload, 2, colSums(aload), "/")

##  START MODELING AND VISUALIZATIONS

library(randomForest)
library(reshape)
library(ggplot2)
library(lda)
library(MASS)

x <- beach_readings

# remove weekends before shifting data 
# as a result, 1 day prior to a Monday will be a Friday
# this is a choice made for modeling to avoid NAs on Mondays (or Tuesdays for 2.days.prior)
x <- x[x$Weekday_code > 1 & x$Weekday_code < 7,]
x <- shift_previous_data(1, x)
x <- shift_previous_data(2, x)

## Begin LDA
## LDA: http://www.r-bloggers.com/computing-and-visualizing-lda-in-r/

## Build out the year-by-year graph
## Each year seems to be different
## Run LDA for each year, get a model for each year
## See how predictors change
## Take out the days that the locks were open. 
## how to deal with binary data?

x_LDA <- x[,c(2,12,15,18,19,101:271)]  #choose predictors
x_LDA <- (x_LDA[,sapply(x_LDA, is.numeric)])# remove non-numerics
na_count <- sapply(x_LDA, function(y) sum(is.na(y))) #analyze NAs
x_LDA <- x_LDA[,na_count < 10000] #enforce NA maximum
x_LDA_complete <- x_LDA[complete.cases(x_LDA),] #remove NAs
x_LDA_complete_scaled <- data.frame(scale(x_LDA_complete)) #scale
x_LDA_complete_scaled$elevated_levels_actual_calculated <- factor(x_LDA_complete$elevated_levels_actual_calculated) #replace scaled with binary
#x_LDA_complete_scaled$elevated_levels_actual_calculated <- factor(x_LDA_complete$'1.daysPrior.CRCW.Lock.Open') #replace scaled with binary
#x_LDA_complete_scaled$elevated_levels_actual_calculated <- factor(x_LDA_complete$'1.daysPrior.Wilmette.Lock.Open') #replace scaled with binary
#x_LDA_complete_scaled$elevated_levels_actual_calculated <- factor(x_LDA_complete$'2.daysPrior.Wilmette.Lock.Open') #replace scaled with binary
x_LDA_complete <- x_LDA_complete[,-c(12,48)]
LDA_results <- lda(elevated_levels_actual_calculated ~ ., x_LDA_complete)
LDA_coefficients <- LDA_results$scaling[order(LDA_results$scaling),]

## End LDA

## MODEL 1
## Try more moving averages to get better weather trends
## https://www.researchgate.net/publication/23955184_Summer_E-coli_Patterns_and_Responses_along_23_Chicago_Beaches

#use the worst 4 beaches
#x1 <- x[x$Client.ID == "Calumet" | x$Client.ID == "63rd" | x$Client.ID == "Montrose" | x$Client.ID == "Rainbow",]

x1 <- x

x_model <- data.frame(x1$e_coli_geomean_actual_calculated,
                      #x1$'2.daysPrior.precipIntensity',
                      #x1$'1.daysPrior.CRCW.Lock.Open',
                      ##x1$'1.daysPrior.humidity',
                      #x1$'2.daysPrior.Day_of_year',
                      #x1$'2.daysPrior.CRCW.Lock.Open',
                      x1$`1.daysPrior.precipIntensity`,
                      x1$`2.daysPrior.precipIntensity`,
                      #x1$Month_code,
                      x1$Beach_code,
                      #x1$Year,
                      #x1$Weekday_code,
                      #x1$'1.daysPrior.Wilmette.Lock.Open',
                      #x1$'1.daysPrior.Day_of_year',
                      #x1$'2.daysPrior.Wilmette.Lock.Open',
                      x1$Full_date)
x_model_complete <- x_model[complete.cases(x_model),] #remove NAs
vars <- length(x_model)

#Loop through years

roc_curve <- data.frame()
#for (year in unique(x_model_complete$x.Year)) {      
for (year in c(2016)) {

  #set.seed(111)
  #ind <- sample(2, nrow(x_model_complete), replace = TRUE, prob=c(0.7, 0.3))
  #x_train <- x_model_complete[ind == 1,]
  #x_test <- x_model_complete[ind == 2,]
  #x_test_epa <- x_small_complete[ind == 2, c(1,3,4,5,7,8)]
  
  #x_train <- x_model_complete[x_model_complete$x.Year %in% setdiff(unique(x_model_complete$x.Year),year),]
  #x_test <- x_model_complete[x_model_complete$x.Year == year,]
  #  x_test_epa <- x_model_by_year_complete[x_model_by_year_complete$x.Year == 2010,]

  x_train <- x_model_complete[x_model_complete$x1.Full_date < "2016-01-01" & x_model_complete$x1.Full_date > "2006-01-01",]
  x_test <- x_model_complete[x_model_complete$x1.Full_date >= "2016-01-01",]

  
  model <- randomForest(x1.e_coli_geomean_actual_calculated ~ ., data=x_train[1:vars-1])
  x_test$prediction <- predict(model, x_test[1:vars-1])
  #model <- randomForest(x1.e_coli_geomean_actual_calculated ~ ., data=x_train)
  #x_test$prediction <- predict(model, x_test)
  x_test$ecoli_binary <- ifelse(x_test$x1.e_coli_geomean_actual_calculated >= 235, 1, 0)
  
  
  true_positive_rates = c()
  false_positive_rates = c()
  for (threshold in seq(0, 750, 1)) {
    x_test$prediction_binary <- ifelse(x_test$prediction >= threshold, 1, 0)
    x_test$true_positive <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 1), 1, 0)
    x_test$true_negative <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 0), 1, 0)
    x_test$false_negative <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 0), 1, 0)
    x_test$false_positive <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 1), 1, 0)
    true_positive_rates = c(true_positive_rates, (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_negative))))
    false_positive_rates = c(false_positive_rates, (sum(x_test$false_positive) / (sum(x_test$false_positive) + sum(x_test$true_negative))))
  }
  
  
  roc_curve_by_year <- data.frame(year, true_positive_rates, false_positive_rates)
  roc_curve <- rbind(roc_curve, roc_curve_by_year)
  
  
#  print(year)
}
roc_curve$year <- as.factor(roc_curve$year)
#roc_curve$year <- 2016
ggplot(data=roc_curve, aes(x=false_positive_rates, y=true_positive_rates, color=year)) + geom_path()
# p <- ggplot()
# p + geom_path(aes(x = false_positive_rates, y = true_positive_rates, color = "My Model")) + geom_vline(xintercept = .05, color = "black") + ylim(0,1) + xlim(0,1)

## MODEL 2
## Using Principal Components (1 - 5) as predictors

## Append PCs to other data for models to work

#x_model <- data.frame(pca$x[,1:5])
#x_model_complete <- x_model[complete.cases(x_model),] #remove NAs


#Loop through years

#roc_curve <- data.frame()
#for (year in unique(x_model_complete$x.Year)) {        
  
#  set.seed(111)
  #  ind <- sample(2, nrow(x_model_complete_2015), replace = TRUE, prob=c(0.7, 0.3))
  #  x_train <- x_model_complete_2015[ind == 1,]
  #  x_test <- x_model_complete_2015[ind == 2,]
  #  x_test_epa <- x_small_complete[ind == 2, c(1,3,4,5,7,8)]
  
#  x_train <- x_model_complete[x_model_complete$x.Year %in% setdiff(unique(x_model_complete$x.Year),year),]
#  x_test <- x_model_complete[x_model_complete$x.Year == year,]
  #  x_test_epa <- x_model_by_year_complete[x_model_by_year_complete$x.Year == 2010,]
  
  
#  model <- randomForest(x.e_coli_geomean_actual_calculated ~ ., data=x_train[1:20])
#  x_test$prediction <- predict(model, x_test[1:20])
#  x_test$ecoli_binary <- ifelse(x_test$x.e_coli_geomean_actual_calculated >= 235, 1, 0)
  
#  true_positive_rates = c()
#  false_positive_rates = c()
#  for (threshold in seq(0, 750, 1)) {
#    x_test$prediction_binary <- ifelse(x_test$prediction >= threshold, 1, 0)
#    x_test$true_positive <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 1), 1, 0)
#    x_test$true_negative <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 0), 1, 0)
#    x_test$false_negative <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 0), 1, 0)
#    x_test$false_positive <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 1), 1, 0)
#    true_positive_rates = c(true_positive_rates, (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_negative))))
#    false_positive_rates = c(false_positive_rates, (sum(x_test$false_positive) / (sum(x_test$false_positive) + sum(x_test$true_negative))))
#  }
  
  
#  roc_curve_by_year <- data.frame(year, true_positive_rates, false_positive_rates)
#  roc_curve <- rbind(roc_curve, roc_curve_by_year)
  
  
#  print(year)
#}
#roc_curve$year <- as.factor(roc_curve$year)
#ggplot(data=roc_curve, aes(x=false_positive_rates, y=true_positive_rates, color=year)) + geom_path()

##  MODEL 3: ROC FOR EPA booster model
##  Only includes the beaches with water sensors in 2016

#x3 <- x[x$Client.ID == "Calumet" | x$Client.ID == "Montrose" | x$Client.ID == "Ohio",]
#x3 <- x[x$Client.ID == "Calumet" | x$Client.ID == "63rd" | x$Client.ID == "Montrose" | x$Client.ID == "Rainbow",]
x3 <- x

#  LDA to choose model predictors

x3_LDA <- x[x$Year == "2015" | x$Year == "2016",c(2,12,15,18,19,101:184)]  #choose predictors for LDA
x3_LDA <- (x3_LDA[,sapply(x3_LDA, is.numeric)])# remove non-numerics
x3_na_count <- sapply(x3_LDA, function(y) sum(is.na(y))) #analyze NAs
x3_LDA <- x3_LDA[,x3_na_count < 1800] #enforce NA maximum
x3_LDA_complete <- x3_LDA[complete.cases(x3_LDA),] #remove NAs
x3_LDA_complete <- x3_LDA_complete[,-c(32,47)]
x3_LDA_results <- lda(elevated_levels_actual_calculated ~ ., data=x3_LDA_complete)
x3_LDA_coefficients <- x3_LDA_results$scaling[order(x3_LDA_results$scaling),]

# build model

#`1.daysPrior.Year`                                       `1.daysPrior.precipIntensity` 
#-5.850576e+01                                            -2.969288e+01 
#`1.daysPrior.Wave.Height.Mean`                           `1.daysPrior.Wave.Height.Min` 
#-7.244513e+00                                            -2.779521e+00 
#`1.daysPrior.Drek_elevated_levels_predicted_calculated`  `1.daysPrior.apparentTemperatureMin` 
#-8.142072e-01                                            -4.931445e-01 
#`1.daysPrior.temperatureMin`                             `1.daysPrior.Interval.Rain.Mean` 
#5.031347e-01                                             6.060437e-01 
#`1.daysPrior.precipIntensityMax`                         `1.daysPrior.moonPhase` 
#6.488877e-01                                             9.923320e-01 
#Drek_elevated_levels_predicted_calculated                `1.daysPrior.humidity` 
#2.833475e+00                                             3.002202e+00 
#`1.daysPrior.Wave.Height.Max` 
#4.544569e+00 


x_small <- data.frame(x3$e_coli_geomean_actual_calculated, 
                      x3$Full_date,
                      x3$Drek_Prediction)
x_model_complete_2 <- x_small[complete.cases(x_small),]
vars <- length(x_small)    

#set.seed(12345)
#ind <- sample(2, nrow(x_model_complete_2), replace = TRUE, prob=c(0.8, 0.2))
#x_train <- x_model_complete_2[x_model_complete_2$x3.Year=="2015",c(1,3:vars-1)]
#x_test <- x_model_complete_2[x_model_complete_2$x3.Year=="2016",c(1,3:vars-1)]
x_test_epa <- x_model_complete_2[x_model_complete_2$x3.Full_date >= "2016-01-01", c(1,vars)] 

model <- randomForest(x_train$x3.e_coli_geomean_actual_calculated ~ ., data=x_train)
x_test$prediction <- predict(model, x_test)
x_test$ecoli_binary <- ifelse(x_test$x3.e_coli_geomean_actual_calculated >= 235, 1, 0)

true_positive_rates = c()
false_positive_rates = c()
for (threshold in seq(0, 1323, 1)) {
  x_test$prediction_binary <- ifelse(x_test$prediction >= threshold, 1, 0)
  x_test$true_positive <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 1), 1, 0)
  x_test$true_negative <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 0), 1, 0)
  x_test$false_negative <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 0), 1, 0)
  x_test$false_positive <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 1), 1, 0)
  true_positive_rates = c(true_positive_rates, (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_negative))))
  false_positive_rates = c(false_positive_rates, (sum(x_test$false_positive) / (sum(x_test$false_positive) + sum(x_test$true_negative))))
  #print(paste0("Threshold = ", threshold, "     TPR = ", 
  #             true_positive_rates[threshold+1], "     FPR = ", 
  #             false_positive_rates[threshold+1]))
}
x_test_epa$ecoli_binary <- ifelse(x_test_epa$x3.e_coli_geomean_actual_calculated >= 235, 1, 0)
epa_true_positive_rates = c()
epa_false_positive_rates = c()
for (threshold in seq(0, 1323, 1)) {
  x_test_epa$Drek_elevated_levels_predicted_calculated <- ifelse(x_test_epa$x3.Drek_Prediction >= threshold, 1, 0)
  x_test_epa$true_positive <- ifelse((x_test_epa$ecoli_binary == 1 & x_test_epa$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
  x_test_epa$true_negative <- ifelse((x_test_epa$ecoli_binary == 0 & x_test_epa$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
  x_test_epa$false_negative <- ifelse((x_test_epa$ecoli_binary == 1 & x_test_epa$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
  x_test_epa$false_positive <- ifelse((x_test_epa$ecoli_binary == 0 & x_test_epa$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
  epa_true_positive_rates <- c(epa_true_positive_rates, (sum(x_test_epa$true_positive) / (sum(x_test_epa$true_positive) + sum(x_test_epa$false_negative))))
  epa_false_positive_rates <- c(epa_false_positive_rates, (sum(x_test_epa$false_positive) / (sum(x_test_epa$false_positive) + sum(x_test_epa$true_negative))))
}
p <- ggplot() 
p + geom_path(aes(x = false_positive_rates, y = true_positive_rates), color = "blue") + geom_path(aes(x = epa_false_positive_rates, y = epa_true_positive_rates), color = "red") + ylim(0,1) + xlim(0,1) + ggtitle("2016 Predictions ROC")
p + geom_path(aes(x = false_positive_rates, y = true_positive_rates), color = "blue") + geom_path(aes(x = epa_false_positive_rates, y = epa_true_positive_rates), color = "red") + ylim(0,.5) + xlim(0,.1) + ggtitle("2016 Prediction ROC")
##  PRECISION / RECALL  ##

precision = c()
recall = c()
for (threshold in seq(0, 1323, 1)) {
  x_test$prediction_binary <- ifelse(x_test$prediction >= threshold, 1, 0)
  x_test$true_positive <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 1), 1, 0)
  x_test$true_negative <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 0), 1, 0)
  x_test$false_negative <- ifelse((x_test$ecoli_binary == 1 & x_test$prediction_binary  == 0), 1, 0)
  x_test$false_positive <- ifelse((x_test$ecoli_binary == 0 & x_test$prediction_binary  == 1), 1, 0)
  precision = c(precision, (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_positive))))
  recall = c(recall, (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_negative))))
  #print(paste0("Threshold = ", threshold, "     Precision = ", 
  #             (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_positive))), "     Recall = ", 
  #             (sum(x_test$true_positive) / (sum(x_test$true_positive) + sum(x_test$false_negative)))))
}

ggplot() + geom_path(aes(x = recall, y = precision), color = "blue") + ylim(0,1) + xlim(0,1)
#ggsave(file="PR-Curve.png")

