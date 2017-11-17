library(ggplot2)
library(data.table)
library(RSocrata)

## for SOLM17 presentation


labs <- read.socrata("https://data.cityofchicago.org/Parks-Recreation/Beach-Lab-Data/2ivx-z93u")
preds <- read.socrata("https://data.cityofchicago.org/Parks-Recreation/Beach-E-coli-Predictions/xvsz-3xcj")
labs$DNA.Sample.Timestamp <- strftime(labs$DNA.Sample.Timestamp, format = "%Y-%m-%d")
df <- merge(preds, labs, by.x = c("Beach.Name", "Date"), by.y = c("Beach", "DNA.Sample.Timestamp"))
dt <- data.table(df)


dtp <- dt[Beach.Name == "Hartigan (Albion)" |
            Beach.Name == "Foster"]
dtp <- dcast(dtp, Date ~ Beach.Name, fun=mean, value.var = c("DNA.Reading.Mean", "Predicted.Level"))
dtp <- na.omit(dtp)
names(dtp)[c(3,5)] <- c("DNA.Reading.Mean_Albion", "Predicted.Level_Albion")
ggplot(dtp, aes(log(DNA.Reading.Mean_Foster), log(DNA.Reading.Mean_Albion))) + 
  geom_point() + 
  geom_smooth(method='lm',formula=y~x, se = FALSE)
ggplot(dtp, aes(log(DNA.Reading.Mean_Foster), log(DNA.Reading.Mean_Albion))) + 
  geom_point() + 
  geom_line(aes(log(dtp$DNA.Reading.Mean_Foster), log(dtp$Predicted.Level_Albion)), color = "blue")

