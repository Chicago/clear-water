source("00_startup.R")
source("10_results.R")


df<-results_df
df$Latitude<-Lat(df$Client.ID)
df$Longitude<-Long(df$Client.ID)
