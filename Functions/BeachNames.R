#This is the function that is used to normalize names of beaches. Throughout the
#years there have been many names for beaches.This functionbrings it down
#to the 20 Beaches we are predicting.

#If you want to add a beach name go to the cleanbeachnames.csv and add the
# needed information to the list.

#Overview of the function:
#You give the function a string or list of strings
#Read in cleanbeachnames.csv
#The beach name is matched up with the shortened beach name
#and is put back out as the shortened beach name.

BeachNames<- function(data){
  
  cleanbeachnames <- read.csv("CSVs/cleanbeachnames.csv", stringsAsFactors = FALSE)
  
  changenames <- setNames(cleanbeachnames$Short_Names, cleanbeachnames$Old)
  
  #delete leading and trailing spaces
  data <- sapply(data, function (x) gsub("^\\s+|\\s+$", "", x))
  
  data <- changenames[data]
  return(data)
}
#This function takes in the beachnames and returns a Latitude
Lat<- function(data){
  cleanbeachnames <- read.csv("CSVs/cleanbeachnames.csv", stringsAsFactors = FALSE)
  
  Lat <- setNames(cleanbeachnames$Latitude, cleanbeachnames$Short_Names )
  
  #delete leading and trailing spaces
  data <- sapply(data, function (x) gsub("^\\s+|\\s+$", "", x))
  
  data<- Lat[data]
  return(data)
}

#This function takes in the beachnames and returns a Longitude
Long<- function(data){
  cleanbeachnames <- read.csv("CSVs/cleanbeachnames.csv", stringsAsFactors = FALSE)
  
  Long <- setNames(cleanbeachnames$Longitude, cleanbeachnames$Short_Names )
  
  #delete leading and trailing spaces
  data <- sapply(data, function (x) gsub("^\\s+|\\s+$", "", x))
  
  data<- Long[data]
  return(data)
}

#This function takes in the beachnames and returns a USGSid
USGSid<- function(data){
  cleanbeachnames <- read.csv("CSVs/cleanbeachnames.csv", stringsAsFactors = FALSE)
  
  Long <- setNames(cleanbeachnames$USGS, cleanbeachnames$Short_Names )
  
  #delete leading and trailing spaces
  data <- sapply(data, function (x) gsub("^\\s+|\\s+$", "", x))
  
  data<- Long[data]
  return(data)
}