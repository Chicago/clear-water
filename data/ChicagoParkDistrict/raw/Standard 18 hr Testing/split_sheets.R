library(plyr)
library(readxl)  
library(psych)
library(dplyr)
library(tidyr)

clean <- function(j, col.names = c(".id","Laboratory.ID","Client.ID","Reading.1","Reading.2","Escherichia.coli","Units","Sample.Collection.Time")) {
  ifelse(ncol(as.data.frame(j[1]))>30, j <- j[2:length(j)], j <- j[1:length(j)])  #get's rid of summary or master sheet 
  j <- lapply(j, function(x) x[-1,, drop=FALSE]) #remove column names due to some errors (especially in 2014 file)
  df <- ldply(j, data.frame) #puts all dfs from the list into one df
  df <- df[1:8]
  names(df)=col.names #add column names in
  df <- df[which(!is.na(df$Client.ID)),] #get's ride of extra rows and ejplainer tejt that appears in the sheets
}

read_excel_allsheets<- function(filename) {
  if (Sys.info()["sysname"] == "Windows") {
    null_file = "NUL"
  } else {
    null_file = "/dev/null"
  }
  
  capture.output(sheets <- readxl::excel_sheets(filename),
                 file = null_file)
  
  capture.output(x <- lapply(sheets, function(X) readxl::read_excel(filename,
                                                                    sheet = X,
                                                                    col_names = F)),
                 file = null_file)
  
  names(x) <- sheets
  x
}

split_sheets <- function(filename, year){
  mysheets <- read_excel_allsheets(filename)
  df <- clean(mysheets)
  df$Year <- year
  df <- df[c(9,1:8)]
  names(df)[names(df) == '.id'] <- 'Date'
  return(df)
}