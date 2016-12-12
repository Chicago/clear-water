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
                 "Sample2.Collection.Time" = results_df$Culture.Sample.2.Timestamp,
                 "DNA.Sample.Timestamp" = results_df$DNA.Sample.Timestamp,
                 "DNA.Reading.1" = results_df$DNA.Sample.1.Reading,
                 "DNA.Reading.2" = results_df$DNA.Sample.2.Reading,
                 "DNA.Geo.Mean" = results_df$DNA.Reading.Mean)

#Split the Date column into Year, Month, and Day columns
results_df$Year <- as.character(results_df$Date, format='%Y')
results_df$Month <- as.character(results_df$Date, format='%m')
results_df$Day <- as.character(results_df$Date, format='%d')


#Normalize the Beach Names
results_df$Client.ID<-as.character(results_df$Client.ID)
results_df$Client.ID<-BeachNames(results_df$Client.ID)

#Bring in 2015 DNA tests
DNA_2015 <- read.csv("CSVs/DNA_2015.csv", stringsAsFactors=FALSE)
names(DNA_2015)[names(DNA_2015)=='South.Shore']<-'South Shore'
names(DNA_2015)[names(DNA_2015)=='X63rd.Street']<-'63rd Street'

#Bring in the Columns and create a line for each column and date
DNA_2015<- gather(DNA_2015,Date, DNA.Geo.Mean)#, Rainbow, South Shore, 63rd Street, Montrose)

#Rename Columns 
colnames(DNA_2015)<-c("Date","Client.ID","DNA.GeoMean")


DNA_2015$Date<-strptime(DNA_2015$Date,format="%m/%d/%Y")
DNA_2015$Year<-strftime(DNA_2015$Date,"%Y")
DNA_2015$Month<-strftime(DNA_2015$Date,format="%m")
DNA_2015$Day<-strftime(DNA_2015$Date,format="%d")

#Normalize the Beach Names
DNA_2015$Client.ID<-as.character(DNA_2015$Client.ID)
DNA_2015$Client.ID<-BeachNames(DNA_2015$Client.ID)


#This for loop goes in finds where the DNA tests were taken according to the 
#main data frame. And puts the DNA mean in
for (i in 1:length(DNA_2015$Year)){
  results_df[which(results_df$Year == DNA_2015$Year[i] &
                    results_df$Month== DNA_2015$Month[i] &
                    results_df$Day == DNA_2015$Day[i] &
                    results_df$Client.ID == DNA_2015$Client.ID[i]),]$DNA.Geo.Mean<-DNA_2015$DNA.GeoMean[i]
}

rm(DNA_2015,i)
