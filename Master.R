source("00_startup.R")
source("01_load.R")

##------------------------------------------------------------------------------
## settings
##------------------------------------------------------------------------------

# predictors
df_model <- df[, c("Escherichia.coli",
                   "Client.ID",
                   "precipProbability",
                   # "Howard_Escherichia.coli",
                   # "63rd_DNA.Geo.Mean",
                   # "South Shore_DNA.Geo.Mean",
                   # "Montrose_DNA.Geo.Mean",
                   # "Calumet_DNA.Geo.Mean", 
                   # "Rainbow_DNA.Geo.Mean",
                   # "Howard_DNA.Geo.Mean",
                   "Date",
                   "Predicted.Level"
)]
model_cols <- (ncol(df_model))
# other settings
trainStart <- "2006-01-01"
trainEnd <- "2014-12-31"
testStart <- "2015-01-01"
testEnd <- "2016-12-31"
excludeBeaches <- c("Rainbow",
                    "South Shore",
                    "Montrose",
                    "Calumet",
                    "63rd",
                    "Howard")
threshBegin <- 0
threshEnd <- 1500
title1 <- "2015-2016 Geomean Model ROC Curve"
title2 <- "2015-2016 Geomean Model ROC Curve"
title3 <- "2015-2016 Geomean Model PR Curve"

source("20_model.R", print.eval=TRUE)

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