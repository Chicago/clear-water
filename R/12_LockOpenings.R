################################################################################
#One of the major areas we want to look at is after the locks are open what does
#that do to the amount of E.coli for the future days. This is the program that
#grabs the days of all the lock openings and gets them ready to add to our main
#df in Master.R
#
#This program reads in the the Lock_opening.csv, finds which locks were actually
#openened and the average volume of water let out per day over the time period
#they were opened. In the end it creates a row for each day the lock was 
#actually opened.
################################################################################

print("Importing Lock Opening Data")
# Read in lock data
lock_data<- read.csv("CSVs/lock_openings.csv", stringsAsFactors=FALSE)
lock_data <- na.omit(lock_data)
lock_data$begin_date <- as.Date(lock_data$begin_date,format="%m/%d/%Y")
lock_data$end_date <- as.Date(lock_data$end_date,format="%m/%d/%Y")

# Create lock indicators for when each lock was open
lock_data$Obrien.Lock.Open <- ifelse(lock_data$O.Brien >0,1,0)
lock_data$CRCW.Lock.Open <- ifelse(lock_data$CRCW>0,1,0)
lock_data$Wilmette.Lock.Open <- ifelse(lock_data$Wilmette>0,1,0)

# Create variable for average volume per day each lock was open
lock_data$Obrien.Lock.Volume <- lock_data$O.Brien / lock_data$days_lock_open
lock_data$CRCW.Lock.Volume <- lock_data$CRCW / lock_data$days_lock_open
lock_data$Wilmette.Lock.Volume <- lock_data$Wilmette / lock_data$days_lock_open

###
#Create a row for every single day the lock was open. Because when the CSV was
#created it was done for a time period rather than every single day.
###


#Create 2 dummy data frames to be able to populate in the for loop. The `Dummy`
#data frame will just be one row gathering all the information that we are
#going to be putting in that row. The test will be all the rows put together
#from the dummy variable.
dummy<-as.data.frame(NULL)
test<-as.data.frame(NULL)

#This for loop will take in every row of the CSV
for(j in 1:length(lock_data$days_lock_open))
{
  #This for loop will be for every day the lock was open on each row of the CSV
  for(i in 1:lock_data$days_lock_open[j])
  {
    #Is a 1x11 dataframe that contains all the columns from lock_data above
    #adding in the specific date we are looking at in the interval.
    dummy<-cbind(lock_data$begin_date[j]-i+2,data.frame(lock_data[j,c("O.Brien",
                                                                      "CRCW",
                                                                      "Wilmette",
                                                                      "total_volume",
                                                                      "Obrien.Lock.Open",
                                                                      "CRCW.Lock.Open",
                                                                      "Wilmette.Lock.Open",
                                                                      "Obrien.Lock.Volume",
                                                                      "CRCW.Lock.Volume",
                                                                      "Wilmette.Lock.Volume")]))
    #This combines all the data in an Nx11 matrix for each individual day the 
    #locks were open.
    test<-rbind(test, dummy)
  }
}

#Rename the Rows cause the came out with weird numbers
rownames(test)<-1:nrow(test)

#Rename the column to Date
names(test)[names(test)=="lock_data$begin_date[j] - i + 2"]<- 'Date'

#Store to the variable that I want to actually work with
lock_data<-test 

#Split the Date column into Year, Month, and Day columns
lock_data$Year <- as.character(lock_data$Date, format='%Y')
lock_data$Month <- as.character(lock_data$Date, format='%m')
lock_data$Day <- as.character(lock_data$Date, format='%d')

#Remove the "Date" column from `lock_data`
lock_data<- lock_data[,!(names(lock_data) %in% c("Date"))]

#Remove the information stored in R that doesn't need to be there anymore
rm(test,dummy,i,j)
