source("data/ChicagoParkDistrict/raw/Standard 18 hr Testing/split_sheets.R")

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

# Combine Data
beach_readings <- rbind(df2006, df2007, df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015)

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

##Merge drek with beach_readings 
beach_readings <- merge(beach_readings, drekdata, by.x = c("Client.ID", "Full_date"), by.y = c("Beach", "Date"), all.x=T)

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


# Calculate confusion matrix in 2015

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


