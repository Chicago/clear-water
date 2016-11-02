df <- read.socrata("https://data.cityofchicago.org/resource/2ivx-z93u.csv")
df<- data.frame( "Date" = df$Culture.Sample.1.Timestamp,
                 "Laboratory.ID" = df$Culture.Test.ID, 
                 "Client.ID" = df$Beach, 
                 "Reading.1" = df$Culture.Sample.1.Reading, 
                 "Reading.2" = df$Culture.Sample.2.Reading,
                 "Escherichia.coli" = df$Culture.Reading.Mean,
                 "Sample.Collection.Time" = df$Culture.Sample.2.Timestamp)
df$Year <- as.character(df$Date, format='%Y')
df$Month <- as.character(df$Date, format='%m')
df$Day <- as.character(df$Date, format='%d')