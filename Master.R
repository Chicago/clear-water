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

# remove prior modeling variables
keep <- list("Beach_Water_Levels", "BeachNames", "df", "Lat", "lock_data", 
             "Long", "modelCurves", "modelEcoli", "results_df", "sourceDir",
             "usePackage", "USGS_predictions_df", "USGSid", "water_quality_df",
             "weather_data")
rm(list=ls()[!ls() %in% keep])

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
# to run without USGS, comment out "Predicted.Level" above and uncomment next line
# df_model$Predicted.Level <- 1 #meaningless value

model_cols <- (ncol(df_model))

#train/test split
kFolds <- TRUE #If TRUE next 4 lines will not be used
trainStart <- "2006-01-01"
trainEnd <- "2011-12-31"
testStart <- "2012-01-01"
testEnd <- "2012-12-31"

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
title1 <- "2015-2016 DNA Model ROC"
title2 <- "2015-2016 USGS Model ROC"
title3 <- "2015-2016 DNA Model PR Curve"
title4 <- "2015-2016 USGS Model PR Curve"

source("30_model.R", print.eval=TRUE)

model_summary <- plot_data %>%
  group_by(thresholds) %>%
  summarize(tpr = mean(tpr),
            fpr = mean(fpr),
            tprUSGS = mean(tprUSGS),
            fprUSGS = mean(fprUSGS),
            precision = mean(precision, na.rm = TRUE),
            recall = mean(recall),
            precisionUSGS = mean(precisionUSGS, na.rm = TRUE),
            recallUSGS = mean(recallUSGS),
            tp = mean(tp),
            fn = mean(fn),
            tn = mean(tn),
            fp = mean(fp),
            tpUSGS = mean(tpUSGS),
            fnUSGS = mean(fnUSGS),
            tnUSGS = mean(tnUSGS),
            fpUSGS = mean(fpUSGS)
            )

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
