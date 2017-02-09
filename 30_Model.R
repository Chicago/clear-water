# takes in settings from Master.R and uses them to 1) create train/test sets
# 2) model and 3) plot curves



model_cols <- (ncol(df_model))


if (kFolds) {
  print("Modeling with 10 folds validation")
  df_model <- df_model[complete.cases(df_model),] #remove NAs from df_model
  set.seed(111)
  dates <- unique(df_model$Date)
  fold_size <- .1 * length(dates)
  dates_sample <- sample(dates, fold_size)
  used_dates <- c()
  plot_data <- data.frame()
  predictions <- data.frame()
  for (fold in c(1:10)) {
    print(paste0("Cross-validating fold # ", fold))
    testDays <- dates_sample
    testData <- df_model[df_model$Date %in% testDays, ]
    testData <- testData[which(!testData$Client.ID %in% excludeBeaches),]
    trainData <- df_model[!df_model$Date %in% testDays, c(1:ncol(df_model) - 1)]
    trainData <- trainData[which(!trainData$Client.ID %in% excludeBeaches),]
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
    }
    model <- modelEcoli(trainData, testData)
    fold_data <- data.frame(fold, 
                            "tpr" = model$tpr,
                            "fpr" = model$fpr,
                            "tprUSGS" = model$tprUSGS,
                            "fprUSGS" = model$fprUSGS,
                            "precision" = model$precision,
                            "recall" = model$recall,
                            "precisionUSGS" = model$precisionUSGS,
                            "recallUSGS" = model$recallUSGS,
                            "tp" = model$tp,
                            "fn" = model$fn,
                            "tn" = model$tn,
                            "fp" = model$fp,
                            "tpUSGS" = model$tpUSGS,
                            "fnUSGS" = model$fnUSGS,
                            "tnUSGS" = model$tnUSGS,
                            "fpUSGS" = model$fpUSGS, 
                            "thresholds" = model$thresholds)
    plot_data <- rbind(plot_data, fold_data)
    predictions <- rbind(predictions, model$predictions)
    used_dates <- c(used_dates, dates_sample)
    remaining_dates <- dates[!dates %in% used_dates]
    if (fold < 10) dates_sample <- sample(remaining_dates, fold_size)
  }
  names(predictions)[names(predictions) == "Predicted.Level"] <- "USGS.Prediction"
  names(predictions)[names(predictions) == "predictionRF"] <- "DNAModel.Prediction"
  plot_data$fold <- as.factor(plot_data$fold)
  plot_data <- plot_data %>%
    group_by(thresholds) %>%
    summarize(tp = sum(tp),
              fn = sum(fn),
              tn = sum(tn),
              fp = sum(fp),
              tpUSGS = sum(tpUSGS),
              fnUSGS = sum(fnUSGS),
              tnUSGS = sum(tnUSGS),
              fpUSGS = sum(fpUSGS)
    )
  plot_data <- mutate(plot_data,
                      tpr = tp/(tp+fn),
                      fpr = fp/(fp+tn),
                      precision = tp/(tp+fp),
                      recall = tp/(tp+fn),
                      tprUSGS = tpUSGS/(tpUSGS+fnUSGS),
                      fprUSGS = fpUSGS/(fpUSGS+tnUSGS),
                      precisionUSGS = tpUSGS/(tpUSGS+fpUSGS),
                      recallUSGS = tpUSGS/(tpUSGS+fnUSGS)
                      )
  p <- ggplot(data = plot_data) 
  print(p + 
          geom_smooth(aes(x = fpr, y = tpr, 
                          color = "DNA Model"),
                      span = .9) + 
          geom_smooth(aes(x = fprUSGS, y = tprUSGS, 
                          color = "USGS Model"),
                      span = .9) + 
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title1))
  print(p + 
          geom_smooth(aes(x = recall, y = precision,
                          color = "DNA Model"),
                      span = .9) +
          geom_smooth(aes(x = recallUSGS, y = precisionUSGS,
                          color = "USGS Model"),
                      span = .9) +
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title2))
} else {
  print("Modeling with user-defined validation data")
  trainData <- df_model[,
                        c(1:model_cols - 1)] #remove EPA prediction from training data
  # Reduce train set to non-predictor beaches
  trainData <- trainData[which(!trainData$Client.ID %in% excludeBeaches),]
  trainData <- trainData[trainData$Date < trainEnd
                         & trainData$Date > trainStart,]
  trainData <- trainData[complete.cases(trainData),] #remove NAs from train data
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
  }
  testData <- df_model[df_model$Date < testEnd
                       & df_model$Date > testStart, ]
  # Reduce test set to non-predictor beaches
  testData <- testData[which(!testData$Client.ID %in% excludeBeaches),]
  testData <- testData[complete.cases(testData),] #remove NAs from test data
  print(paste0("Train set observations = ",nrow(trainData)))
  print(paste0("Test set observations = ",nrow(testData)))
  model <- modelEcoli(trainData, testData)
  p <- ggplot() 
  print(p + 
          geom_smooth(aes(x = model$fpr, y = model$tpr, 
                          color = "DNA Model"), 
                      span = .9) + 
          geom_smooth(aes(x = model$fprUSGS, y = model$tprUSGS, 
                          color = "USGS Model"),
                      span = .9) + 
          ylim(0,1) + 
          xlim(0,1) + 
          ggtitle(title1))
  print(p + 
          geom_smooth(aes(x = model$recall, y = model$precision,
                          color = "DNA Model"),
                      span = .9) +
          geom_smooth(aes(x = model$recallUSGS, y = model$precisionUSGS,
                          color = "USGS MOdel"),
                      span = .9) +
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title2))
  plot_data <- as.data.frame(model)
}
