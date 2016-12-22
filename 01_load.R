source("10_LabResults.R")
source("11_USGSpredictions.R")
source("12_LockOpenings.R")
source("13_Beach_Water_Levels.R")
source("14_Weather.R")
source("15_WaterQuality.R")

#Create the main Data Frame as `DF`
df<-results_df[!is.na(results_df$Client.ID),]


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

#Merge in the Water Level data from 13_Beach_Water_Levels.R
df<-merge(df,Beach_Water_Levels,by.x=c("Year","Month","Day"),
          by.y=c("Year","Month","Day"),all.x=TRUE)

#Merge in the Weather data from 14_Weather.R
df<-merge(df,weather_data,by.x=c("Year","Month","Day","Client.ID"),
          by.y=c("Year","Month","Day","beach"),all.x=TRUE)

#Merge in the Water Quality data from 15_WaterQuality.R
df<-merge(df,water_quality_df,by.x=c("Year","Month","Day","Client.ID"),
          by.y=c("Year","Month","Day","Client.ID"),all.x=TRUE)

#This function finds the days that have multiple readings for a beach, and pulls
#out the specific dates that it happened on.
beach_dup<-unique(df[duplicated(df[,c("Year","Month","Day","Client.ID")]),c("Year","Month","Day")])

#The following for loop goes through the days that have duplicates in them, 
#Finds the duplicated beaches, takes the geometric mean of all the duplicated beaches
#and puts the answer in the original spot.
for(i in 1:length(beach_dup$Year)){
  #Pull in the observations on the days that are duplicated
  daily_beach_obs<-df[which(df$Year==beach_dup[i,"Year"] &
                              df$Month==beach_dup[i,"Month"] & 
                              df$Day==beach_dup[i,"Day"]),]
  #Find the all the beaches that have multiple observations on a given day
  mult_obs_beach<-unique(daily_beach_obs[duplicated(daily_beach_obs[1:4]),]$Client.ID)
  #This is in case there are multiple duplicates on different beaches, so we can get
  #All of them read on a given day
  for(j in 1:length(mult_obs_beach)){
    #Grab a beach that has multiple observations
    check<-daily_beach_obs[daily_beach_obs$Client.ID==mult_obs_beach[j],]
    #Use an if else statement in case there is a day with all observations being N/A
    if ( is.na(prod(as.numeric(check$Escherichia.coli,na.rm=TRUE)))){
      df[row.names(check)[1],]$Escherichia.coli<-NA
    }
    else{
      #Produce a geomean of all the duplicated beaches and assign it to the first
      #beach in a datafr),]
      df[row.names(check)[1],]$Escherichia.coli <-prod(as.numeric(check$Escherichia.coli,na.rm=TRUE))^(1/length(check$Escherichia.coli))
    }
  }
}

#Remove the dataframes that were created from the memory
rm(beach_dup,daily_beach_obs,mult_obs_beach,check,drop_list,i,j)

#Get rid of the duplicated rows
df<-df[!duplicated(df[,1:4]),]

#add new columns for predictor beaches.
#We will use the same-day test results for these beaches to predict other beaches
for (beach in unique(df$Client.ID)) {
  df <- addLabsColumn(df, beach, "Escherichia.coli")
}
for (beach in unique(df$Client.ID)) {
  df <- addLabsColumn(df, beach, "DNA.Geo.Mean")
}
