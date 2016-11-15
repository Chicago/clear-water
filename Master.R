source("00_startup.R")
source("10_results.R")
source("11_USGSpredictions.R")
source("12_LockOpenings.R")
source("14_Beach_Water_Levels.R")

#Create the main Data Frame as `DF`
df<-results_df

#Add Latitude and Longitude to `DF`
df$Latitude<-Lat(df$Client.ID)
df$Longitude<-Long(df$Client.ID)

#The parks department predicts the levels of E.coli at 9 beaches and then
#extends those predictions out to the beaches around those 9 beaches.
#The USGSid is what beaches are getting which predictions. The predictions can
#be found on the City of Chicago's website.
df$USGS<-USGSid(df$Client.ID)

#Merge in the USGS Predictions from 11_USGSpredictions.R
df<-merge(df,USGS_predictions_df,by.x=c("USGS","Year","Month","Day"),
          by.y=c("USGS","Year","Month","Day"),all.x=TRUE)

#Merge in the Lock data from 12_LockOpening.R
df<-merge(df,lock_data,by.x=c("Year","Month","Day"),
          by.y=c("Year","Month","Day"),all.x=TRUE)

#Merge in the Lock data from 14_Beach_Water_Levels.R
df<-merge(df,Beach_Water_Levels,by.x=c("Year","Month","Day"),
          by.y=c("Year","Month","Day"),all.x=TRUE)

