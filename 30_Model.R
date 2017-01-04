# takes in settings from Master.R and uses them to 1) create train/test sets
# 2) model and 3) plot curves

print("Modeling Data")

if (kFolds) {
  set.seed(111)
  daysOfYear <- unique(df_model$DayOfYear)
  testDays <- sample(daysOfYear, size = .1 * length(daysOfYear))
  testData <- df_model[df_model$DayOfYear %in% testDays, ]
  trainData <- df_model[!df_model$DayOfYear %in% testDays, c(1:ncol(df_model) - 1)]
  train_vars <- ncol(trainData)
  test_vars <- ncol(testData)
} else {
  
  ##------------------------------------------------------------------------------
  ## train 
  ##------------------------------------------------------------------------------
  
  trainData <- df_model[,
                        c(1:model_cols - 1)] #remove EPA prediction from training data
  # Reduce train set to non-predictor beaches
  trainData <- trainData[which(!trainData$Client.ID %in% excludeBeaches),]
  trainData <- trainData[trainData$Date < trainEnd
                         & trainData$Date > trainStart,]
  trainData <- trainData[complete.cases(trainData),] #remove NAs from train data
  train_vars <- ncol(trainData)
  
  if (downsample) {
    train_high <- trainData[trainData$Escherichia.coli >= highMin
                            & trainData$Escherichia.coli < highMax, ]
    train_low <- trainData[trainData$Escherichia.coli < lowMax, ]
    # only use as many low days as you have high days
    ind <- sample(c(1:nrow(train_low)),
                  nrow(train_high),
                  replace = TRUE)
    train_balanced <- rbind(train_high, train_low[ind, ])
    trainData <- train_balanced
    rm(list = c("train_high",
                "train_low",
                "ind",
                "train_balanced",
                "highMin",
                "highMax",
                "lowMax"
    )
    )
  }
  
  ##------------------------------------------------------------------------------
  ## test
  ##------------------------------------------------------------------------------
  
  testData <- df_model[df_model$Date < testEnd
                       & df_model$Date > testStart, ]
  # Reduce test set to non-predictor beaches
  testData <- testData[which(!testData$Client.ID %in% excludeBeaches),]
  testData <- testData[complete.cases(testData),] #remove NAs from test data
  test_vars <- ncol(testData)
  
}

##------------------------------------------------------------------------------
## modeling / curves / result pair (add to dataframe?)
##------------------------------------------------------------------------------

model <- randomForest(Escherichia.coli ~ .,
                      data = trainData[,
                                       c(1:(train_vars - 2))])
testData$predictionRF <- predict(model, testData[,c(1:(test_vars-3))])

tpr <- c()
fpr <- c()
tprUSGS <- c()
fprUSGS <- c()
precision <- c()
recall <- c()
precisionUSGS <- c()
recallUSGS <- c()
testData$actual_binary <- ifelse(testData$Escherichia.coli >= 235, 1, 0)
for (threshold in seq(threshBegin, threshEnd, 1)) {
  testData$predictionRF_binary <- ifelse(testData$predictionRF >= threshold, 1, 0)
  testData$USGS_binary <- ifelse(testData$Predicted.Level >= threshold, 1, 0)
  testData$true_positive <- ifelse((testData$actual_binary == 1 & testData$predictionRF_binary  == 1), 1, 0)
  testData$true_negative <- ifelse((testData$actual_binary == 0 & testData$predictionRF_binary  == 0), 1, 0)
  testData$false_negative <- ifelse((testData$actual_binary == 1 & testData$predictionRF_binary  == 0), 1, 0)
  testData$false_positive <- ifelse((testData$actual_binary == 0 & testData$predictionRF_binary  == 1), 1, 0)
  testData$true_positiveUSGS <- ifelse((testData$actual_binary == 1 & testData$USGS_binary  == 1), 1, 0)
  testData$true_negativeUSGS <- ifelse((testData$actual_binary == 0 & testData$USGS_binary  == 0), 1, 0)
  testData$false_negativeUSGS <- ifelse((testData$actual_binary == 1 & testData$USGS_binary  == 0), 1, 0)
  testData$false_positiveUSGS <- ifelse((testData$actual_binary == 0 & testData$USGS_binary  == 1), 1, 0)
  tpr = c(tpr, (sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_negative))))
  fpr = c(fpr, (sum(testData$false_positive) / (sum(testData$false_positive) + sum(testData$true_negative))))
  tprUSGS <- c(tprUSGS, (sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_negativeUSGS))))
  fprUSGS <- c(fprUSGS, (sum(testData$false_positiveUSGS) / (sum(testData$false_positiveUSGS) + sum(testData$true_negativeUSGS))))
  precision = c(precision, (sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_positive))))
  recall = c(recall, (sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_negative))))
  precisionUSGS <- c(precisionUSGS, (sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_positiveUSGS))))
  recallUSGS <- c(recallUSGS, (sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_negativeUSGS))))
}

#use following if looping k-folds
#roc_curve_by_fold <- data.frame(fold, tpr, fpr)
#roc_curve <- rbind(roc_curve, roc_curve_by_year)
#roc_curve_by_fold <- data.frame(fold, tpr, fpr)
#ggplot(data=roc_curve, aes(x=fpr, y=tpr, color=fold)) + geom_path()

p <- ggplot() 
p + 
  geom_path(aes(x = fpr, y = tpr), 
            color = "blue") + 
  geom_path(aes(x = fprUSGS, y = tprUSGS), 
            color = "red") + 
  ylim(0,1) + 
  xlim(0,1) + 
  ggtitle(title1)
p + 
  geom_path(aes(x = fpr, y = tpr), 
            color = "blue") + 
  geom_path(aes(x = fprUSGS, y = tprUSGS), 
            color = "red") + 
  ylim(0,.75) + 
  xlim(0,.1) + 
  ggtitle(title2)
p + 
  geom_path(aes(x = recall, y = precision),
            color = "blue") +
  geom_path(aes(x = recallUSGS, y = precisionUSGS),
            color = "red") +
  ylim(0,1) + 
  xlim(0,1) +
  ggtitle(title3)

## cleanup after modelings
rm(list=c("df_model",
          "downsample",
          "fpr",
          "fprUSGS",
          "kFolds",
          "model",
          "model_cols",
          "p",
          "precision",
          "precisionUSGS",
          "recall",
          "recallUSGS",
          "test_vars",
          "testData",
          "threshold",
          "tpr",
          "tprUSGS",
          "train_vars",
          "trainData",
          "trainStart",
          "trainEnd",
          "testStart",
          "testEnd",
          "excludeBeaches",
          "threshBegin",
          "threshEnd",
          "title1",
          "title2",
          "title3"
))