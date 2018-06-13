################################################################################
#2016 & Future Results
################################################################################

print("Importing USGS Prediction Data")
USGS_predictions_df<-read.socrata("https://data.cityofchicago.org/resource/t62e-8nvc.csv")

USGS_predictions_df$Beach.Name<-BeachNames(USGS_predictions_df$Beach.Name)

#Split the Date column into Year, Month, and Day columns
USGS_predictions_df$Year <- as.character(USGS_predictions_df$Date, format='%Y')
USGS_predictions_df$Month <- as.character(USGS_predictions_df$Date, format='%m')
USGS_predictions_df$Day <- as.character(USGS_predictions_df$Date, format='%d')

#Take out The Date column
USGS_predictions_df<- USGS_predictions_df[,!(names(USGS_predictions_df) %in% c("Date","RecordID"))]
USGS_predictions_df$USGS<-USGSid(USGS_predictions_df$Beach.Name)

PredictorBeaches<-unique(USGS_predictions_df$Beach.Name)
################################################################################
#2015 Results
################################################################################

#Read in 2015 from a CSV
drek <- read.csv("CSVs/daily_summaries_drekb.csv",stringsAsFactors=FALSE)

#Delete the beaches that have no meaning/ carry the same information from the
#CSV that was read in
drop_list<-c("Lane-Beach","Loyola-Beach","Marion-Mahoney-Griffin-Beach",
             "Columbia-Beach","Hartigan-Beach","Tobey-Prinz-Beach")

drek<-drek[!(drek$Beach %in% drop_list),]


#Clean the Beach Names in `drek`
drek$Beach<-BeachNames(drek$Beach)
#Rename the 'Beach'column in `drek`
names(drek)[names(drek)=='Beach']<-'Beach.Name'
#The parks department predicts the levels of E.coli at 9 beaches and then
#extends those predictions out to the beaches around those 9 beaches.
#The USGSid is what beaches are getting which predictions.Using the 
#PredictorBeaches variable from above, we are only going to use the observations
#from the 9 predicted beaches in the `drek table`
drek<-drek[drek$Beach.Name %in% PredictorBeaches,]

#In order to use the rbind() later the "Reading" column is not needed, but a
#"Probability column is needed. The "Reading" column name was changed to
#"Probability" and then all the values were turned NA. The "Prediction" 
#column name was changed to "Predicted.Level" to match the same
#Column name in USGS_predictions_df for rbind() later.
names(drek)[names(drek)=='Reading']<-'Probability'
drek$Probability<-NA
names(drek)[names(drek)=='Prediction']<-'Predicted.Level'

#The "SwimStatus" column name was changed to "Swim.Advisory"
#to match the same Column name in USGS_predictions_df for rbind() later.
#Also the levels were changed in the "Swim.Advisory" column to match the
#levels that were in the USGS_predictions_df.
names(drek)[names(drek)=='SwimStatus']<-'Swim.Advisory'
drek$Swim.Advisory <- ifelse(drek$Swim.Advisory=='No Restrictions','N','Y')

#Make the date column the Date type
drek$Date<-as.Date(drek$Date,"%m/%d/%Y")
#Split the Date column into Year, Month, and Day columns
drek$Year <- as.character(drek$Date, format='%Y')
drek$Month <- as.character(drek$Date, format='%m')
drek$Day <- as.character(drek$Date, format='%d')

#Remove the DateColumn
drek<- drek[,!(names(drek) %in% c("Date"))]

#Add on the USGSid to `drek`
drek$USGS<-USGSid(drek$Beach.Name)

#remove NAs
USGS_predictions_df <- na.omit(USGS_predictions_df)

USGS_predictions_df[USGS_predictions_df$Predicted.Level < 0,
                    c("Predicted.Level","Probability")]<-0

#Combine the data from `USGS_predictions_df` and `drek` into one data frame
USGS_predictions_df<-rbind(USGS_predictions_df,drek)

rm(drek)
rm(PredictorBeaches)

