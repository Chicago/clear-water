#In this file we are just getting the observations from each day from 2006-Present
#The means are all online, bring them in put them in a data frame and rename the
#columns of the data frame for easy identification. Normalize the Beach Names
#so that we can combine with other things in the future.

results_df <- read.socrata("https://data.cityofchicago.org/resource/2ivx-z93u.csv")
results_df<- data.frame( "Date" = results_df$Culture.Sample.1.Timestamp,
                 "Laboratory.ID" = results_df$Culture.Test.ID, 
                 "Client.ID" = results_df$Beach, 
                 "Reading.1" = results_df$Culture.Sample.1.Reading, 
                 "Reading.2" = results_df$Culture.Sample.2.Reading,
                 "Escherichia.coli" = results_df$Culture.Reading.Mean,
                 "Sample.Collection.Time" = results_df$Culture.Sample.2.Timestamp)

#Split the Date column into Year, Month, and Day columns
results_df$Year <- as.character(results_df$Date, format='%Y')
results_df$Month <- as.character(results_df$Date, format='%m')
results_df$Day <- as.character(results_df$Date, format='%d')


#Normalize the Beach Names
results_df$Client.ID<-as.character(results_df$Client.ID)
results_df$Client.ID<-BeachNames(results_df$Client.ID)
