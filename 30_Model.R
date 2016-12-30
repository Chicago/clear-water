# takes in settings from Master.R and uses them to 1) create train/test sets
# 2) model and 3) plot curves

print("Modeling Data")

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

# trainData <- trainData[!trainData$Year == year,]
#  train_high <- trainData[trainData$e_coli_geomean_actual_calculated >= 200 
#                      & trainData$e_coli_geomean_actual_calculated < 2500,]
#  train_low <- trainData[trainData$e_coli_geomean_actual_calculated < 200,]
# only use as many low days as you have high days
#  ind <- sample(c(1:nrow(train_low)), 
#                nrow(train_high), 
#                replace = TRUE)
#  train_balanced <- rbind(train_high, train_low[ind,])
#  trainData <- train_balanced

## the following will produce a random split
## this will replace everthing done with test/train above
## comment out if you want to use the above code
## there may be errors as this is old pasted code

#set.seed(111)
#data_split <- df_model[complete.cases(df_model),]
#even_days <- data_split[data_split$Day_of_year %% 2 == 0,]
#odd_days <- data_split[data_split$Day_of_year %% 2 == 1,]
#ind_even <- sample(2, nrow(even_days), replace = TRUE, prob=c(0.5, 0.5))
#ind_odd <- sample(2, nrow(odd_days), replace = TRUE, prob=c(0.5, 0.5))
#test_even <- even_days[ind_even == 2,]
#test_odd <- odd_days[ind_odd == 2,] 
#train_even <- even_days[ind_even == 1,c(1:model_cols-1)] #remove EPA prediction from training data
#train_odd <- odd_days[ind_odd == 1,c(1:model_cols-1)] #remove EPA prediction from training data
#testData <- rbind(test_even, test_odd)
#trainData <- rbind(train_even, train_odd)
#train_vars <- ncol(trainData)
#test_vars <- ncol(test)

##------------------------------------------------------------------------------
## test
##------------------------------------------------------------------------------

testData <- df_model[df_model$Date < testEnd
                     & df_model$Date > testStart, ]
testData <- testData[which(!testData$Client.ID %in% excludeBeaches),]
testData <- testData[complete.cases(testData),] #remove NAs from test data
test_vars <- ncol(testData)

#testData <- df_model[df_model$Year == year,]
#testData <- df_model[df_model$Year == year,
#                    c(1:model_cols-1)]
# Reduce test set to non-predictor beaches

##------------------------------------------------------------------------------
## modeling / curves / result pair (add to dataframe?)
##------------------------------------------------------------------------------

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

#use following if looping over years
#roc_curve_by_year <- data.frame(year, tpr, fpr)
#roc_curve <- rbind(roc_curve, roc_curve_by_year)
#roc_curve_by_year <- data.frame(year, tpr, fpr)
#ggplot(data=roc_curve, aes(x=fpr, y=tpr, color=year)) + geom_path()

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
          "fpr",
          "fprUSGS",
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