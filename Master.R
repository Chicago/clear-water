source("00_startup.R")
source("10_results.R")
source("11_USGSpredictions.R")


df<-results_df
df$Latitude<-Lat(df$Client.ID)
df$Longitude<-Long(df$Client.ID)
df$USGS<-USGSid(df$Client.ID)
df<-merge(df,USGS_predictions_df,by.x=c("USGS","Year","Month","Day"),by.y=c("USGS","Year","Month","Day"),all.x=TRUE)

