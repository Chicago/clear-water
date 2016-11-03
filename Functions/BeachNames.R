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
  
  cleanbeachnames <- read.csv("cleanbeachnames.csv", stringsAsFactors = FALSE)
  
  changenames <- setNames(cleanbeachnames$Short_Names, cleanbeachnames$Old) 
  data <- sapply(data, function (x) gsub("^\\s+|\\s+$", "", x)) #delete leading and trailing spaces
  
  data <- changenames[data]
  return(data)
}