#-------------------------------------------------------------------------------
#  CLEAR WATER: Predicting Water Quality in Chicago Beaches
#
#  All user-defined settings are found in this file
#  Make changes below as described to manipulate the model
#  The model and evaluation code is located in 30_Model.R and Functions/modelEColi.R
#
#  Run this file only - all other code is pulled in by Master.R
#-------------------------------------------------------------------------------

# Load libraries and functions
source("R/00_Startup.R")

#-------------------------------------------------------------------------------
#  Ingest Data
#-------------------------------------------------------------------------------

# The following .R files have been run already and are cached in Data/df.Rds

# source("R/10_LabResults.R")
# source("R/11_USGSpredictions.R")
# source("R/12_LockOpenings.R")
# source("R/13_Beach_Water_Levels.R")
# source("R/14_Weather.R")
# source("R/15_WaterQuality.R")
# source("R/20_Clean.R")
# saveRDS(df, paste0(getwd(),"/Data/df.Rds"))

df <- readRDS(paste0(getwd(),"/Data/df.Rds"))

# remove prior modeling variables before starting up a new model
# keep <- list("df", "modelCurves", "modelEcoli")
# rm(list=ls()[!ls() %in% keep])

#-------------------------------------------------------------------------------
#  CHOOSE PREDICTORS
#  Comment out the predictors that you do not want to use
#-------------------------------------------------------------------------------

# set predictors
df_model <- df[, c("Escherichia.coli", #dependent variable
                   "Client.ID",

                   "Foster_Escherichia.coli",
                   "North_Avenue_Escherichia.coli",
                   "n31st_Escherichia.coli",
                   "Leone_Escherichia.coli",
                   "South_Shore_Escherichia.coli",
                   
                   "Year", #used for splitting data
                   "Date" #used for splitting data
)]

finaltest <- df_model[df_model$Year == "2016",]

#-------------------------------------------------------------------------------
#  CHOOSE TEST/TRAIN SETS
#  You can decide whether to use kFolds cross validation or define your own sets
#  If you set kFolds to TRUE, the data will be separated into 10 sets
#  If you set kFolds to FALSE, the model will use trainStart, trainEnd, etc. (see below)
#  CANNOT BE USED IF productionMode = TRUE
#-------------------------------------------------------------------------------

kFolds <- FALSE #If TRUE next 2 lines will not be used but cannot be commented out
testYears <- c("2016")
trainYears <- c("2006", "2007", "2008", "2009","2010", "2011", "2012", "2013", "2014", "2015")
# trainYears <- trainYears[! trainYears %in% testYears]

# If productionMode is set to TRUE, a file named model.Rds will be generated
# Its used is explained at https://github.com/Chicago/clear-water-app
# Set trainYears to what you would like the model to train on
# testYears must still be specified, although not applicable
# plots will not be accurate

productionMode <- FALSE

#-------------------------------------------------------------------------------
#  DOWNSAMPLING
#  If you set downsample to TRUE, choose the 3 variables below
#  The training set will be a 50/50 split of 1) data less than the "lowMax" and
#  2) data between the "highMin" and "highMax"
#-------------------------------------------------------------------------------

# downsample settings
downsample <- FALSE #If FALSE comment out the next 3 lines
highMin <- 235
highMax <- 2500
lowMax <- 235


#-------------------------------------------------------------------------------
#  EXCLUDE ENTIRE BEACHES FROM THE TEST SET
#  This is important if you use same-day beach test results as a predictor
#  If so, the predictor beach should not be a beach that is being predicted
#  because the model would then be predicting on data it was trained on.
#  Comment out any beach that you used as a predictor.
#-------------------------------------------------------------------------------

excludeBeaches <- c(
                    # "12th",
                    "31st",
                    # "39th",
                    # "57th",
                    "63rd",
                    # "Albion",
                    "Calumet",
                    "Foster",
                    # "Howard",
                    # "Jarvis",
                    # "Juneway",
                    "Leone",
                    "Montrose",
                    "North Avenue",
                    # "Oak Street",
                    "Ohio",
                    # "Osterman",
                    "Rainbow",
                    # "Rogers",
                    "South Shore"
                    )

#-------------------------------------------------------------------------------
#  NAME PLOTS
#  These are automatically generated based on the settings chosen above
#-------------------------------------------------------------------------------

title1 <- paste0("ROC", 
                 if(kFolds == TRUE) " - kFolds",
                 if(kFolds == FALSE) " - validate on ",
                 if(kFolds == FALSE) testYears)
title2 <- paste0("PR Curve", 
                 if(kFolds == TRUE) " - kFolds",
                 if(kFolds == FALSE) " - validate on ",
                 if(kFolds == FALSE) testYears)


#-------------------------------------------------------------------------------
#  THRESHHOLD
#  These settings can be used to manipulate the plots and the model_summary dataframe
#-------------------------------------------------------------------------------

threshBegin <- 1
threshEnd <- 1000


thresh <- 235

#-------------------------------------------------------------------------------
#  RUN MODEL
#  Plots will generate and results will be saved in "model_summary"
#-------------------------------------------------------------------------------

# runs all modeling code
source("R/30_Model.R", print.eval=TRUE)

# creates a data frame with all model results
# this aggregates the folds to generate one single curve
# for user-defined test set, this doesn't have any effect
model_summary <- plot_data %>%
  group_by(thresholds) %>%
  summarize(tpr = mean(tpr),
            fpr = mean(fpr),
            precision = mean(precision, na.rm = TRUE),
            recall = mean(recall),
            tp = mean(tp),
            fn = mean(fn),
            tn = mean(tn),
            fp = mean(fp)
  )

saveRDS(model, paste0("models/", "model-hybrid-only-2016-holdout", ".Rds"))

