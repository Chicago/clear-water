beaches <- c(# "12th",
  # "31st",
  # "39th",
  # "57th",
  "63rd",
  # "Albion",
  #"Calumet",
  # "Foster",
  #"Howard",
  # "Jarvis",
  # "Juneway",
  # "Leone",
  #"Montrose",
  # "North Avenue",
  # "Oak Street",
  # "Ohio",
  # "Osterman",
  #"Rainbow",
  # "Rogers",
  "South Shore")
beach_choose <- function(beaches){ 

  source("00_startup.R",local=TRUE)
  
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
  #rm(list=ls()[!ls() %in% keep])
  
  # set predictors
  
beach_to_colnames <- setNames(c("12th_Escherichia.coli","31st_Escherichia.coli" , "57th_Escherichia.coli", "n63rd_Escherichia.coli", "Albion_Escherichia.coli", "Calumet_Escherichia.coli", "Foster_Escherichia.coli", "Howard_Escherichia.coli", "Jarvis_Escherichia.coli", "Juneway_Escherichia.coli", "Leone_Escherichia.coli", "Montrose_Escherichia.coli", "North Avenue_Escherichia.coli","Oak Street_Escherichia.coli", "Ohio_Escherichia.coli", "Osterman_Escherichia.coli", "Rainbow_Escherichia.coli", "Rogers_Escherichia.coli", "South_Shore_Escherichia.coli", "39th_Escherichia.coli"),c("12th","31st","57th", "63rd", "Albion", "Calumet", "Foster", "Howard", "Jarvis", "Juneway","Leone", "Montrose", "North Avenue", "Oak Street", "Ohio", "Osterman", "Rainbow", "Rogers", "South Shore", "39th"))
beaches_rename <- as.character(beach_to_colnames[beaches])

df_model <- df[, unlist(list(c("Escherichia.coli","Client.ID"),beaches_rename,
                     c("Date", #Must use for splitting data, not included in model
                     "Predicted.Level" #Must use for USGS model comparison, not included in model
  )))]
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
  excludeBeaches <- beaches
  
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
  
  # change threshold for saving results into "predictions" data frame
  thresh <- 235
  
  # runs all modeling code
  source("30_model.R", print.eval=TRUE, local = TRUE)
  
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
  return(model_summary)
}
