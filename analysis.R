prCurve <- function(truth, predicted_values) {
    recalls = c()
    precisions = c()
    for (i in 0:20) {
        recalls[i+1] = recall(truth, predicted_values >= i * 100)
        precisions[i+1] = precision(truth, predicted_values >= i * 100)
    }
    
    lines(recalls ~ precisions)
}


recall <- function(truth, predict) {
    return(sum(predict[truth])/sum(truth))
}

precision <- function(truth, predict) {
    return(sum(predict[truth])/sum(predict))
}

measures <- read.csv('data/daily_summaries_drekb.csv')

measures$Date <- as.Date(measures$Date, '%m/%d/%Y')

measures$tomorrow <- measures$Date + 1

measures <- merge(measures, measures[, !(names(measures) %in% c("Date"))],
                  by.x=c('Beach', 'Date'),
                  by.y=c('Beach', 'tomorrow'))

measures <- measures[,c(1,2,3,4,5,8,9)]

names(measures) <- c("beach", "date", "reading", "prediction", "status",
                     "yesterday_reading", "yesterday_prediction")

true_ban_days <- measures$reading > 1000

plot(c(0,1), c(0,1), type="n")

prCurve(true_ban_days,  measures$yesterday_reading)

prCurve(true_ban_days,  measures$prediction)

model.naive <- glm(reading ~ yesterday_reading*beach + weekdays(date)*beach, measures, family='poisson')
summary(model.naive)

prCurve(true_ban_days,  exp(predict(model.naive)))

model.forecast <- glm(reading ~ prediction*beach + weekdays(date)*beach + months(date), measures, family='poisson')
summary(model.forecast)

prCurve(true_ban_days,  exp(predict(model.forecast)))


