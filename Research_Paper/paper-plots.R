library(ggplot2)
library(data.table)
library(RSocrata)
library(ROCR)
library(heatmaply)
library(corrplot)

## read in data

labs <- data.table(read.socrata("https://data.cityofchicago.org/Parks-Recreation/Beach-Lab-Data/2ivx-z93u"))
labs$Date <- strftime(labs$Culture.Sample.1.Timestamp, format = "%Y-%m-%d")

## create graph showing how beaches move together

labsPartial <- labs[Date > as.Date("2016-07-01") & Date < as.Date("2016-07-31")]
ggplot(labsPartial, aes(Date, log(Culture.Reading.Mean), group = Beach)) + 
  geom_jitter(height = 0, width = .05, aes(color = Beach, stroke = .75)) +
  geom_hline(aes(yintercept = log(235)), linetype = "dashed") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

## grab hybrid model ROC data

source("Master.R")

## create beach correlation plots

dt <- data.table(df)
beachCor <- dcast(dt, Date ~ Client.ID, fun.aggregate = mean, value.var = "Escherichia.coli")
beachCor <- na.omit(beachCor)
beachCor <- log(beachCor[,c(2:21)])
corTable <- cor(beachCor)
corTable <- round(corTable, 2)
heatmaply_cor(corTable)
corrplot(corTable, method = "circle")

## build USGS ROC

dfSelected <- df[df$Client.ID %in% c(
  "12th",
  "39th",
  "57th",
  "Albion",
  "Howard",
  "Jarvis",
  "Juneway",
  "Oak Street",
  "Osterman",
  "Rogers"
),]

usgs <- dfSelected[!is.na(dfSelected$Escherichia.coli) & 
                     !is.na(dfSelected$Predicted.Level),
                   c("Escherichia.coli","Predicted.Level")]
names(usgs) <- c("actual","predicted")
usgs$actualBin <- ifelse(usgs$actual < 235, 0, 1)

pred <- prediction(usgs$predicted, usgs$actualBin)
perf <- performance(pred, "tpr", "fpr")
usgsROC <- data.frame(fpr = unlist(perf@x.values),
                       tpr = unlist(perf@y.values))

## plot hybrid and usgs ROC together

ggplot() + 
  geom_line(aes(x = model$fpr, y = model$tpr, color = "Hybrid Model")) + 
  geom_line(aes(x = usgsROC$fpr, y = usgsROC$tpr, color = "Prior-day Model")) +
  ylim(0,1) + 
  xlim(0,1) + 
  xlab("False Positive Rate") +
  ylab("True Positive Rate") +
  geom_vline(xintercept = .018, linetype = "dashed") +
  scale_colour_manual("", 
                      breaks = c("Hybrid Model", "Prior-day Model"),
                      values = c("blue", "red")) 
