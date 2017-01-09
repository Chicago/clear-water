#This function is used to run a Random Forest model and return
#TPR, FPR, recall, and precision
modelEcoli <- function(trainData, testData) {
  train_vars <- ncol(trainData)
  test_vars <- ncol(testData)
  model <- randomForest(Escherichia.coli ~ .,
                        data = trainData[,
                                         c(1:(train_vars - 1))])
  testData$predictionRF <- predict(model, testData[,c(1:(test_vars-2))])
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
  list("tpr"=tpr,
       "fpr"=fpr,
       "tprUSGS"=tprUSGS,
       "fprUSGS"=fprUSGS,
       "precision"=precision,
       "recall"=recall,
       "precisionUSGS"=precisionUSGS,
       "recallUSGS"=recallUSGS)
}