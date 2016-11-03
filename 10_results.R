#In this file we are just getting the observations from each day from 2006-Present
#The means are all online, bring them in put them in a data frame and rename the
#columns of the data frame for easy identification. Normalize the Beach Names
#so that we can combine with other things in the future.

df <- read.socrata("https://data.cityofchicago.org/resource/2ivx-z93u.csv")
df<- data.frame( "Date" = df$Culture.Sample.1.Timestamp,
                 "Laboratory.ID" = df$Culture.Test.ID, 
                 "Client.ID" = df$Beach, 
                 "Reading.1" = df$Culture.Sample.1.Reading, 
                 "Reading.2" = df$Culture.Sample.2.Reading,
                 "Escherichia.coli" = df$Culture.Reading.Mean,
                 "Sample.Collection.Time" = df$Culture.Sample.2.Timestamp)

#Split the Date column into Year, Month, and Day columns
df$Year <- as.character(df$Date, format='%Y')
df$Month <- as.character(df$Date, format='%m')
df$Day <- as.character(df$Date, format='%d')

#Normalize the Beach Names
df$Client.ID<-as.character(df$Client.ID)
df$Client.ID<-BeachNames(df$Client.ID)
