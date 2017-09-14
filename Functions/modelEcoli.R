#This function is used to run a Random Forest model and return
#TPR, FPR, recall, and precision
modelEcoli <- function(trainData, testData, threshBegin, threshEnd, thresh, productionMode) {
  train_vars <- ncol(trainData)
  test_vars <- ncol(testData)
  model <- randomForest(Escherichia.coli ~ .,
                        data = trainData[,
                                         c(1:(train_vars - 1))],
                        ntree = 64)
  testData$predictionRF <- predict(model, testData[,c(1:(test_vars-2))])
  tp <- c()
  fn <- c()
  tn <- c()
  fp <- c()
  tpUSGS <- c()
  fnUSGS <- c()
  tnUSGS <- c()
  fpUSGS <- c()
  tpr <- c()
  fpr <- c()
  tprUSGS <- c()
  fprUSGS <- c()
  precision <- c()
  recall <- c()
  precisionUSGS <- c()
  recallUSGS <- c()
  thresholds <- c()
  predictions <- data.frame()
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
    tp <- c(tp, sum(testData$true_positive))
    fn <- c(fn, sum(testData$false_negative))
    tn <- c(tn, sum(testData$true_negative))
    fp <- c(fp, sum(testData$false_positive))
    tpUSGS <- c(tpUSGS, sum(testData$true_positiveUSGS))
    fnUSGS <- c(fnUSGS, sum(testData$false_negativeUSGS))
    tnUSGS <- c(tnUSGS, sum(testData$true_negativeUSGS))
    fpUSGS <- c(fpUSGS, sum(testData$false_positiveUSGS))
    tpr = c(tpr, sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_negative)))
    fpr = c(fpr, sum(testData$false_positive) / (sum(testData$false_positive) + sum(testData$true_negative)))
    tprUSGS <- c(tprUSGS, sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_negativeUSGS)))
    fprUSGS <- c(fprUSGS, sum(testData$false_positiveUSGS) / (sum(testData$false_positiveUSGS) + sum(testData$true_negativeUSGS)))
    precision = c(precision, sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_positive)))
    recall = c(recall, sum(testData$true_positive) / (sum(testData$true_positive) + sum(testData$false_negative)))
    precisionUSGS <- c(precisionUSGS, sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_positiveUSGS)))
    recallUSGS <- c(recallUSGS, sum(testData$true_positiveUSGS) / (sum(testData$true_positiveUSGS) + sum(testData$false_negativeUSGS)))
    thresholds <- c(thresholds, threshold)
    if (threshold == thresh) {
      predictions <- rbind(predictions, testData)
    }
  }
  if (productionMode) {
    print("Saving model.Rds in your working directory")
    saveRDS(model, "model.Rds")
  }
  list("tpr"=tpr,
       "fpr"=fpr,
       "tprUSGS"=tprUSGS,
       "fprUSGS"=fprUSGS,
       "precision"=precision,
       "recall"=recall,
       "precisionUSGS"=precisionUSGS,
       "recallUSGS"=recallUSGS,
       "tp"=tp,
       "fn"=fn,
       "tn"=tn,
       "fp"=fp,
       "tpUSGS"=tpUSGS,
       "fnUSGS"=fnUSGS,
       "tnUSGS"=tnUSGS,
       "fpUSGS"=fpUSGS,
       "thresholds"=thresholds,
       "predictions"=predictions)
}
