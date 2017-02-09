source("00_startup.R")

# source("10_LabResults.R")
# source("11_USGSpredictions.R")
# source("12_LockOpenings.R")
# source("13_Beach_Water_Levels.R")
# source("14_Weather.R")
# source("15_WaterQuality.R")
# source("20_Clean.R")
# saveRDS(df, paste0(getwd(),"/Data/df.Rds"))
df <- readRDS(paste0(getwd(),"/Data/df.Rds"))


##------------------------------------------------------------------------------
## MODEL SETTINGS
##   Make changes to the settings below to tweak the model
##   These settings determine how the model is built inside 30_Model.R
##   The model itself is in /Functions/modelEcoli.R
##------------------------------------------------------------------------------

# remove prior modeling variables before starting up model
keep <- list("df", "modelCurves", "modelEcoli")
rm(list=ls()[!ls() %in% keep])

# set predictors
df_model <- df[, c("Escherichia.coli", #dependent variable
                   "Client.ID",
                   # "precipProbability",
                   # "Water.Level",
                   "Howard_Escherichia.coli",
                   # "n57th_Escherichia.coli", 
                   # "n63rd_Escherichia.coli",
                   # "South_Shore_Escherichia.coli",
                   # "Montrose_Escherichia.coli",
                   # "Calumet_Escherichia.coli",
                   # "Rainbow_Escherichia.coli",
                   # "Ohio_DNA.Geo.Mean",
                   # "North_Avenue_DNA.Geo.Mean",
                   "n63rd_DNA.Geo.Mean",
                   "South_Shore_DNA.Geo.Mean",
                   "Montrose_DNA.Geo.Mean",
                   "Calumet_DNA.Geo.Mean",
                   "Rainbow_DNA.Geo.Mean",
                   "Date", #Must use for splitting data, not included in model
                   "Predicted.Level" #Must use for USGS model comparison, not included in model
                   )]
# to run without USGS for comparison, comment out "Predicted.Level" above and uncomment next line
# df_model$Predicted.Level <- 1 #meaningless value

# train/test data
kFolds <- TRUE #If TRUE next 4 lines will not be used but cannot be commented out
trainStart <- "2006-01-01"
trainEnd <- "2015-12-31"
testStart <- "2016-01-01"
testEnd <- "2016-12-31"

# downsample settings
downsample <- FALSE #If FALSE comment out the next 3 lines
# highMin <- 200
# highMax <- 2500
# lowMax <- 200

# these beaches will not be in test data
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

# change title names for plots
title1 <- paste0("ROC", 
                 if(kFolds == TRUE) " - kFolds",
                 if(kFolds == FALSE) " - validate on ",
                 if(kFolds == FALSE) testStart,
                 if(kFolds == FALSE) " to ",
                 if(kFolds == FALSE) testEnd)
title2 <- paste0("PR Curve", 
                 if(kFolds == TRUE) " - kFolds",
                 if(kFolds == FALSE) " - validate on ",
                 if(kFolds == FALSE) testStart,
                 if(kFolds == FALSE) " to ",
                 if(kFolds == FALSE) testEnd)


# change threshold range for curve plots -- this is the E. Coli value for issuing a swim advisory
threshBegin <- 1
threshEnd <- 500

# runs all modeling code
source("30_model.R", print.eval=TRUE)

# creates a data frame with all model results
# this aggregates the folds to generate one single curve
# for user-defined test set, this doesn't have any effect
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