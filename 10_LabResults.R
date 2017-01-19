#In this file we are just getting the observations from each day from 2006-Present
#The means are all online, bring them in put them in a data frame and rename the
#columns of the data frame for easy identification. Normalize the Beach Names
#so that we can combine with other things in the future.

print("Importing E. Coli Lab Data")
results_df <- read.socrata("https://data.cityofchicago.org/resource/2ivx-z93u.csv")
results_df<- data.frame( "Date" = results_df$Culture.Sample.1.Timestamp,
                 "Laboratory.ID" = results_df$Culture.Test.ID, 
                 "Client.ID" = results_df$Beach, 
                 "Reading.1" = results_df$Culture.Sample.1.Reading, 
                 "Reading.2" = results_df$Culture.Sample.2.Reading,
                 "Escherichia.coli" = results_df$Culture.Reading.Mean,
                 "Sample2.Collection.Time" = results_df$Culture.Sample.2.Timestamp,
                 "DNA.Sample.Timestamp" = results_df$DNA.Sample.Timestamp,
                 "DNA.Reading.1" = results_df$DNA.Sample.1.Reading,
                 "DNA.Reading.2" = results_df$DNA.Sample.2.Reading,
                 "DNA.Geo.Mean" = results_df$DNA.Reading.Mean)

#Split the Date column into Year, Month, and Day columns
results_df$Year <- as.character(results_df$Date, format='%Y')
results_df$Month <- as.character(results_df$Date, format='%m')
results_df$Day <- as.character(results_df$Date, format='%d')

#Add in the the number of day it is in the year, day of the week
results_df$DayOfYear<-strftime(results_df$Date, format = '%j')
results_df$DayOfWeek<-strftime(results_df$Date, format = '%A')


#Normalize the Beach Names
results_df$Client.ID<-as.character(results_df$Client.ID)
results_df$Client.ID<-BeachNames(results_df$Client.ID)

#Remove Beaches that don't have Eschercia.Coli readings
results_df<-results_df[!is.na(results_df$Escherichia.coli),]

#There are a couple outliers in Reading.2 so we will reassign it to 2420.0 and 
#Replace the geometric mean in Eshcheria.Coli
results_df[!is.na(results_df$Reading.2) 
           & results_df$Reading.2>2420,"Reading.2"]<-2420.0

#Geometric Mean Code
results_df$Escherichia.coli<-sqrt(apply(results_df[c("Reading.1","Reading.2")],
                                        1,prod,na.rm=TRUE))

