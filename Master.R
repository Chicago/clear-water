source("00_startup.R")
source("10_LabResults.R")
source("11_USGSpredictions.R")
source("12_LockOpenings.R")
source("13_Beach_Water_Levels.R")
source("14_Weather.R")
source("15_WaterQuality.R")
source("20_Clean.R")

##------------------------------------------------------------------------------
## Model settings
##------------------------------------------------------------------------------

# predictors
df_model <- df[, c("Escherichia.coli",
                   "Client.ID",
                   "precipProbability",
                   "Water.Level",
                   "Howard_Escherichia.coli",
                   # "n57th_Escherichia.coli",
                   # "n63rd_Escherichia.coli", 
                   # # "South_Shore_Escherichia.coli",
                   # "Montrose_Escherichia.coli",
                   # "Calumet_Escherichia.coli",
                   # "Rainbow_Escherichia.coli",
                   "n63rd_DNA.Geo.Mean",
                   "South_Shore_DNA.Geo.Mean",
                   "Montrose_DNA.Geo.Mean",
                   "Calumet_DNA.Geo.Mean",
                   "Rainbow_DNA.Geo.Mean",
                   "Date", #Must use for splitting data
                   "Predicted.Level" #Must use for USGS model comparison
                   )]
model_cols <- (ncol(df_model))

#train/test split
kFolds <- TRUE #If TRUE next 4 lines will not be used
trainStart <- "2016-01-01"
trainEnd <- "2016-07-31"
testStart <- "2016-08-01"
testEnd <- "2016-12-31"

#downsample settings
downsample <- FALSE #If FALSE comment out the next 3 lines
# highMin <- 200
# highMax <- 2500
# lowMax <- 200

excludeBeaches <- c(
                    # "12th",
                    # "31st",
                    # "39th",
                    # "57th",
                    "63rd",
                    # "Albion",
                    "Calumet",
                    # "Foster",
                    "Howard",
                    # "Jarvis",
                    # "Juneway",
                    # "Leone",
                    "Montrose",
                    # "North Avenue",
                    # "Oak Street",
                    # "Ohio",
                    # "Osterman",
                    "Rainbow",
                    # "Rogers",
                    "South Shore"
                    )
threshBegin <- 1
threshEnd <- 1500
title1 <- "2015-2016 DNAmean Model ROC Curve"
title2 <- "2015-2016 DNAmean Model ROC Curve"
title3 <- "2015-2016 DNAmean Model PR Curve"

source("30_model.R", print.eval=TRUE)

## decide whether to keep the analysis below

#-----------------------------------------------------------------------------------------------------------------
# Look at Genetic Tests (need to change variable names)
#-----------------------------------------------------------------------------------------------------------------

#dna <- df[,c(1:15)] # remove culture test columns
#summary(dna)
#plot(dna$DNA.Sample.1.Reading, dna$DNA.Sample.2.Reading)
#plot(dna$DNA.Sample.1.Reading, dna$DNA.Sample.2.Reading, log=c('x', 'y'))
#plot(log(dna$DNA.Sample.1.Reading)+1, log(dna$DNA.Sample.2.Reading)+1, log=c('x', 'y'))
#plot(dna$Escherichia.coli, dna$DNA.Reading.Mean)
#plot(dna$Escherichia.coli, dna$DNA.Reading.Mean, log=c('x', 'y'))
#plot(log(dna$Escherichia.coli)+1, log(dna$DNA.Reading.Mean)+1, log=c('x', 'y'))
#llmodel <- lm(log(log(Escherichia.coli)+1)~log(log(dna$DNA.Reading.Mean)+1), data=dna)
#summary(llmodel)
#par(mfrow=c(2,2));plot(llmodel);par(mfrow=c(1,1))
#hist(dna$DNA.Reading.Mean)

#-----------------------------------------------------------------------------------------------------------------
# Calculate USGS Confusion Matrix (need to change variable names)
#-----------------------------------------------------------------------------------------------------------------

# df_2015 <- beach_readings[beach_readings$Year == "2015",]
# df_2015 <- df_2015[!is.na(df_2015$Drek_elevated_levels_predicted_calculated),]
# df_2015 <- df_2015[!is.na(df_2015$elevated_levels_actual_calculated),]
# tp <- ifelse((df_2015$elevated_levels_actual_calculated == 1 & df_2015$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
# tn <- ifelse((df_2015$elevated_levels_actual_calculated == 0 & df_2015$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
# fn <- ifelse((df_2015$elevated_levels_actual_calculated == 1 & df_2015$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
# fp <- ifelse((df_2015$elevated_levels_actual_calculated == 0 & df_2015$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
# print(paste0("True Positives = ", sum(tp)))
# print(paste0("True Negatives = ", sum(tn)))
# print(paste0("False Positives = ", sum(fp)))
# print(paste0("False Negatives = ", sum(fn)))
# print(paste0("2015 True Positive Rate = ",(sum(tp)/(sum(tp)+sum(fn)))))
# print(paste0("2015 False Positive Rate = ",(sum(fp)/(sum(fp)+sum(tn)))))
# 
# df_2016 <- beach_readings[beach_readings$Year == "2016",]
# df_2016 <- df_2016[!is.na(df_2016$Drek_elevated_levels_predicted_calculated),]
# df_2016 <- df_2016[!is.na(df_2016$elevated_levels_actual_calculated),]
# tp <- ifelse((df_2016$elevated_levels_actual_calculated == 1 & df_2016$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
# tn <- ifelse((df_2016$elevated_levels_actual_calculated == 0 & df_2016$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
# fn <- ifelse((df_2016$elevated_levels_actual_calculated == 1 & df_2016$Drek_elevated_levels_predicted_calculated  == 0), 1, 0)
# fp <- ifelse((df_2016$elevated_levels_actual_calculated == 0 & df_2016$Drek_elevated_levels_predicted_calculated  == 1), 1, 0)
# print(paste0("True Positives = ", sum(tp)))
# print(paste0("True Negatives = ", sum(tn)))
# print(paste0("False Positives = ", sum(fp)))
# print(paste0("False Negatives = ", sum(fn)))
# print(paste0("2016 True Positive Rate = ",(sum(tp)/(sum(tp)+sum(fn)))))
# print(paste0("2016 False Positive Rate = ",(sum(fp)/(sum(fp)+sum(tn)))))