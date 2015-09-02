recall <- function(truth, predict) {
    return(sum(predict[truth])/sum(truth))
}

precision <- function(truth, predict) {
    return(sum(predict[truth])/sum(predict))
}

measures <- read.csv('data/daily_summaries_drekb.csv')

measures$Date <- as.Date(measures$Date, '%m/%d/%Y')

measures$tomorrow <- measures$Date + 1

measures <- merge(measures, measures,
                  by.x=c('Beach', 'Date'),
                  by.y=c('Beach', 'tomorrow'))

measures <- measures[,c(1,2,3,4,5,8,9)]

names(measures) <- c("beach", "date", "reading", "prediction", "status",
                     "yesterday_reading", "yesterday_prediction")

recall(measures$reading > 1000, measures$yesterday_reading > 500)
precision(measures$reading > 1000, measures$yesterday_reading > 500)

recall(measures$reading > 1000, measures$prediction > 200)
precision(measures$reading > 1000, measures$prediction > 200)

model.naive <- glm(reading ~ yesterday_reading*beach, measures, family='poisson')
summary(model.naive)

recall(measures$reading > 1000, exp(predict(model.naive)) > 300)
precision(measures$reading > 1000, exp(predict(model.naive)) > 300)



model.forecast <- glm(reading ~ prediction*beach, measures, family='poisson')
summary(model.forecast)

recall(measures$reading > 1000, exp(predict(model.forecast)) > 700)
precision(measures$reading > 1000, exp(predict(model.forecast)) > 700)


