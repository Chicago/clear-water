# takes in settings from Master.R and uses them to 1) create train/test sets
# 2) model and 3) plot curves

model_cols <- (ncol(df_model))
set.seed(111)

if (kFolds & !productionMode) {
  print("Modeling with 10 folds validation")
  df_model <- df_model[complete.cases(df_model),] #remove NAs from df_model
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
    trainData <- df_model[!df_model$Date %in% testDays, c(1:(ncol(df_model) - 2))]
    trainData <- trainData[which(!trainData$Client.ID %in% excludeBeaches),]
    if (downsample) {
      train_high <- trainData[trainData$Escherichia.coli >= highMin
                              & trainData$Escherichia.coli < highMax, ]
      train_low <- trainData[trainData$Escherichia.coli < lowMax, ]
      # use 1:5 ratio of high days to total days
      ind <- sample(c(1:nrow(train_low)),
                    nrow(train_high) * 4,
                    replace = TRUE)
      train_balanced <- rbind(train_high, train_low[ind, ])
      trainData <- train_balanced
    }
    model <- modelEcoli(trainData, testData, threshBegin, threshEnd, thresh, productionMode)
    fold_data <- data.frame(fold, 
                            "tpr" = model$tpr,
                            "fpr" = model$fpr,
                            "precision" = model$precision,
                            "recall" = model$recall,
                            "tp" = model$tp,
                            "fn" = model$fn,
                            "tn" = model$tn,
                            "fp" = model$fp,
                            "thresholds" = model$thresholds)
    plot_data <- rbind(plot_data, fold_data)
    predictions <- rbind(predictions, model$predictions)
    used_dates <- c(used_dates, dates_sample)
    remaining_dates <- dates[!dates %in% used_dates]
    if (fold < 10) dates_sample <- sample(remaining_dates, fold_size)
  }
  names(predictions)[names(predictions) == "predictionRF"] <- "DNAModel.Prediction"
  plot_data$fold <- as.factor(plot_data$fold)
  plot_data <- plot_data %>%
    group_by(thresholds) %>%
    summarize(tp = sum(tp),
              fn = sum(fn),
              tn = sum(tn),
              fp = sum(fp)
    )
  plot_data <- mutate(plot_data,
                      tpr = tp/(tp+fn),
                      fpr = fp/(fp+tn),
                      precision = tp/(tp+fp),
                      recall = tp/(tp+fn)
                      )
  p <- ggplot(data = plot_data) 
  print(p + 
          geom_line(aes(x = fpr, y = tpr, 
                          color = "DNA Model")) + 
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title1))
  print(p + 
          geom_line(aes(x = recall, y = precision,
                          color = "DNA Model")) +
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title2))
} else {
  print("Modeling with user-defined validation data")
  trainData <- df_model[, c(1:model_cols)]
  # Reduce train set to non-predictor beaches
  trainData <- trainData[which(!trainData$Client.ID %in% excludeBeaches),]
  trainData <- trainData[trainData$Year %in% trainYears,]
  trainData <- trainData[complete.cases(trainData),] #remove NAs from train data
  if (downsample) {
    train_high <- trainData[trainData$Escherichia.coli >= highMin
                            & trainData$Escherichia.coli < highMax, ]
    train_low <- trainData[trainData$Escherichia.coli < lowMax, ]
    # use 1:5 ratio of high days to total days
    ind <- sample(c(1:nrow(train_low)),
                  nrow(train_high) * 4,
                  replace = TRUE)
    train_balanced <- rbind(train_high, train_low[ind, ])
    trainData <- train_balanced
  }
  testData <- df_model[df_model$Year %in% testYears, ]
  # Reduce test set to non-predictor beaches
  testData <- testData[which(!testData$Client.ID %in% excludeBeaches),]
  testData <- testData[complete.cases(testData),] #remove NAs from test data
  print(paste0("Train set observations = ",nrow(trainData)))
  print(paste0("Test set observations = ",nrow(testData)))
  model <- modelEcoli(trainData, testData, threshBegin, threshEnd, thresh, productionMode)
  p <- ggplot() 
  print(p + 
          geom_line(aes(x = model$fpr, y = model$tpr, 
                          color = "DNA Model")) + 
          ylim(0,1) + 
          xlim(0,1) + 
          ggtitle(title1))
  print(p + 
          geom_line(aes(x = model$recall, y = model$precision,
                          color = "DNA Model")) +
          ylim(0,1) + 
          xlim(0,1) +
          ggtitle(title2))
  plot_data <- as.data.frame(model[-which(names(model) == "predictions" | names(model) == "model")])
}
