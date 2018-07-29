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

# Transform wind variables for modeling

# df$windDirectionMath <- 270 - df$windBearing
# df$windU <- df$windSpeed * cos(df$windDirectionMath)
# df$windV <- df$windSpeed * sin(df$windDirectionMath)
# 
# df$windDirectionMath_hourly_1 <- 270 - df$windBearing_hourly_1
# df$windU_hourly_1 <- df$windSpeed_hourly_1 * cos(df$windDirectionMath_hourly_1)
# df$windV_hourly_1 <- df$windSpeed_hourly_1 * sin(df$windDirectionMath_hourly_1)
# df$windDirectionMath_hourly_2 <- 270 - df$windBearing_hourly_2
# df$windU_hourly_2 <- df$windSpeed_hourly_2 * cos(df$windDirectionMath_hourly_2)
# df$windV_hourly_2 <- df$windSpeed_hourly_2 * sin(df$windDirectionMath_hourly_2)
# df$windDirectionMath_hourly_3 <- 270 - df$windBearing_hourly_3
# df$windU_hourly_3 <- df$windSpeed_hourly_3 * cos(df$windDirectionMath_hourly_3)
# df$windV_hourly_3 <- df$windSpeed_hourly_3 * sin(df$windDirectionMath_hourly_3)
# df$windDirectionMath_hourly_4 <- 270 - df$windBearing_hourly_4
# df$windU_hourly_4 <- df$windSpeed_hourly_4 * cos(df$windDirectionMath_hourly_4)
# df$windV_hourly_4 <- df$windSpeed_hourly_4 * sin(df$windDirectionMath_hourly_4)
# df$windDirectionMath_hourly_5 <- 270 - df$windBearing_hourly_5
# df$windU_hourly_5 <- df$windSpeed_hourly_5 * cos(df$windDirectionMath_hourly_5)
# df$windV_hourly_5 <- df$windSpeed_hourly_5 * sin(df$windDirectionMath_hourly_5)
# df$windDirectionMath_hourly_6 <- 270 - df$windBearing_hourly_6
# df$windU_hourly_6 <- df$windSpeed_hourly_6 * cos(df$windDirectionMath_hourly_6)
# df$windV_hourly_6 <- df$windSpeed_hourly_6 * sin(df$windDirectionMath_hourly_6)
# df$windDirectionMath_hourly_7 <- 270 - df$windBearing_hourly_7
# df$windU_hourly_7 <- df$windSpeed_hourly_7 * cos(df$windDirectionMath_hourly_7)
# df$windV_hourly_7 <- df$windSpeed_hourly_7 * sin(df$windDirectionMath_hourly_7)
# df$windDirectionMath_hourly_8 <- 270 - df$windBearing_hourly_8
# df$windU_hourly_8 <- df$windSpeed_hourly_8 * cos(df$windDirectionMath_hourly_8)
# df$windV_hourly_8 <- df$windSpeed_hourly_8 * sin(df$windDirectionMath_hourly_8)
# 
# df_shift_1 <- shift_previous_data(1, df)
# df_shift_2 <- shift_previous_data(2, df)
# df_shift_3 <- shift_previous_data(3, df)
# 
# df <- cbind(df, df_shift_1[,584:1080])
# df <- cbind(df, df_shift_2[,584:1080])
# df <- cbind(df, df_shift_3[,584:1080])
# 
# saveRDS(df, paste0(getwd(),"/Data/df-3-day.Rds"))

df <- readRDS(paste0(getwd(),"/Data/df-3-day.Rds"))

#-------------------------------------------------------------------------------
#  ADD   PREDICTORS
#-------------------------------------------------------------------------------

df$precipIntensity.3.day.total <- df$precipIntensity.1.daysPrior +
  df$precipIntensity.2.daysPrior +
  df$precipIntensity.3.daysPrior

df$precipIntensity.by.8am <- df$precipIntensity_hourly_1 +
  df$precipIntensity_hourly_2 +
  df$precipIntensity_hourly_3 +
  df$precipIntensity_hourly_4 +
  df$precipIntensity_hourly_5 +
  df$precipIntensity_hourly_6 +
  df$precipIntensity_hourly_7 +
  df$precipIntensity_hourly_8 

df$cloudCover.3.day.total <- df$cloudCover.1.daysPrior +
  df$cloudCover.2.daysPrior +
  df$cloudCover.3.daysPrior

df$sunlightTime <- df$sunsetTime - df$sunriseTime

df$windSpeed.3.day.total <- df$windSpeed.1.daysPrior +
  df$windSpeed.2.daysPrior +
  df$windSpeed.3.daysPrior

df$windSpeed.by.8am <- df$windSpeed_hourly_1 +
  df$windSpeed_hourly_2 +
  df$windSpeed_hourly_3 +
  df$windSpeed_hourly_4 +
  df$windSpeed_hourly_5 +
  df$windSpeed_hourly_6 +
  df$windSpeed_hourly_7 +
  df$windSpeed_hourly_8

df$windU.3.day.total <- df$windU.1.daysPrior +
  df$windU.2.daysPrior +
  df$windU.3.daysPrior

df$windU.by.8am <- df$windU_hourly_1 +
  df$windU_hourly_2 +
  df$windU_hourly_3 +
  df$windU_hourly_4 +
  df$windU_hourly_5 +
  df$windU_hourly_6 +
  df$windU_hourly_7 +
  df$windU_hourly_8

df$windV.3.day.total <- df$windV.1.daysPrior +
  df$windV.2.daysPrior +
  df$windV.3.daysPrior

df$windV.by.8am <- df$windV_hourly_1 +
  df$windV_hourly_2 +
  df$windV_hourly_3 +
  df$windV_hourly_4 +
  df$windV_hourly_5 +
  df$windV_hourly_6 +
  df$windV_hourly_7 +
  df$windV_hourly_8

df$Water.Level.3.day.total <- df$Water.Level.1.daysPrior +
  df$Water.Level.1.daysPrior +
  df$Water.Level.1.daysPrior

df$DayOfWeek <- as.factor(df$DayOfWeek)

df$Obrien.Lock.Volume.3.day.total <- df$Obrien.Lock.Volume.1.daysPrior +
  df$Obrien.Lock.Volume.2.daysPrior +
  df$Obrien.Lock.Volume.3.daysPrior

df$CRCW.Lock.Volume.3.day.total <- df$CRCW.Lock.Volume.1.daysPrior +
  df$CRCW.Lock.Volume.2.daysPrior +
  df$CRCW.Lock.Volume.3.daysPrior

df$Wilmette.Lock.Volume.3.day.total <- df$Wilmette.Lock.Volume.1.daysPrior +
  df$Wilmette.Lock.Volume.2.daysPrior +
  df$Wilmette.Lock.Volume.3.daysPrior

#-------------------------------------------------------------------------------
#  CHOOSE PREDICTORS
#  Comment out the predictors that you do not want to use
#-------------------------------------------------------------------------------

# set predictors
df_model <- df[, c("Escherichia.coli", #dependent variable
                   "Client.ID", #beach name
                   
                   ## Precipitation

                   "precipProbability",
                   "precipIntensity.1.daysPrior",
                   "precipIntensity.3.day.total",
                   "precipIntensity.by.8am",
                   
                   ## Sunlight
                   
                   "cloudCover.1.daysPrior",
                   "cloudCover.3.day.total",
                   "sunlightTime",
                   
                   ## Wind
              
                   "windSpeed.1.daysPrior",
                   "windSpeed.3.day.total",
                   "windSpeed.by.8am",
                   "windU.1.daysPrior",
                   "windU.3.day.total",
                   "windU.by.8am",
                   "windV.1.daysPrior",
                   "windV.3.day.total",
                   "windV.by.8am",
                   
                   ## Tidal levels
                   
                   "moonPhase",
                   
                   ## Lake levels
                   
                   "Water.Level",
                   "Water.Level.1.daysPrior",
                   "Water.Level.3.day.total",
                   
                   ## Density of humans and animals
                   
                   "DayOfWeek",
                   "DayOfYear",
                   
                   ### Variables NOT cited in our paper from prior literature
                   
                   ## Lock openings
                   
                   "Obrien.Lock.Volume.1.daysPrior",
                   "CRCW.Lock.Volume.1.daysPrior",
                   "Wilmette.Lock.Volume.1.daysPrior",
                   "Obrien.Lock.Volume.3.day.total",
                   "CRCW.Lock.Volume.3.day.total",
                   "Wilmette.Lock.Volume.3.day.total",
                  
                   ## Today's readings at selected beaches
                   
                   # "Foster_Escherichia.coli",
                   # "North_Avenue_Escherichia.coli",
                   # "n31st_Escherichia.coli",
                   # "Leone_Escherichia.coli",
                   # "South_Shore_Escherichia.coli",
                   
                   ## Train/Test split data

                   "Year", 
                   "Date" 
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

saveRDS(model, paste0("models/", "model-other-only-2016-holdout", ".Rds"))


