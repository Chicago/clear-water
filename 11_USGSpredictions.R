USGS_predictions_df<-read.socrata("https://data.cityofchicago.org/resource/t62e-8nvc.csv")

USGS_predictions_df$Beach.Name<-BeachNames(USGS_predictions_df$Beach.Name)

#Split the Date column into Year, Month, and Day columns
USGS_predictions_df$Year <- as.character(USGS_predictions_df$Date, format='%Y')
USGS_predictions_df$Month <- as.character(USGS_predictions_df$Date, format='%m')
USGS_predictions_df$Day <- as.character(USGS_predictions_df$Date, format='%d')

#Take out The Date column
USGS_predictions_df<- USGS_predictions_df[,!(names(USGS_predictions_df) %in% c("Date","RecordID"))]
USGS_predictions_df$USGS<-USGSid(USGS_predictions_df$Beach.Name)
