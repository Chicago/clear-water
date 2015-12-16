#Chicago Park District Raw Data Cleaning

#set working directory


#Slow way, but with dates -- 2006 through 2015
library(plyr)
library(readxl)  
library(psych)
library(dplyr)
library(tidyr)

col.names=c(".id","Laboratory.ID","Client.ID","Reading.1","Reading.2","Escherichia.coli","Units","Sample.Collection.Time")                 

clean <- function(j) {
      ifelse(ncol(as.data.frame(j[1]))>30, j <- j[2:length(j)], j <- j[1:length(j)])  #get's rid of summary or master sheet 
      j <- lapply(j, function(x) x[-1,, drop=FALSE]) #remove column names due to some errors (especially in 2014 file)
      df <- ldply(j, data.frame) #puts all dfs from the list into one df
      df <- df[1:8]
      names(df)=col.names #add column names in
      df <- df[which(!is.na(df$Client.ID)),] #get's ride of extra rows and ejplainer tejt that appears in the sheets
}

read_excel_allsheets<- function(filename) {  
      sheets <- readxl::excel_sheets(filename)
      x <-    lapply(sheets, function(X) readxl::read_excel(filename, sheet = X, col_names = F))
      names(x) <- sheets
      x
}

mysheets <- read_excel_allsheets("./2006 Lab Results.xls")
df=clean(mysheets)
df2006 <- df
df2006$year <- 2006

mysheets <- read_excel_allsheets("./2007 Lab Results.xls")
df=clean(mysheets)
df2007 <- df
df2007$year <- 2007

mysheets <- read_excel_allsheets("./2008 Lab Results.xls")
df=clean(mysheets)
df2008 <- df
df2008$year <- 2008

mysheets <- read_excel_allsheets("./2009 Lab Results.xls")
df=clean(mysheets)
df2009 <- df
df2009$year <- 2009

mysheets <- read_excel_allsheets("./2010 Lab Results.xls")
df=clean(mysheets)
df2010 <- df
df2010$year <- 2010

mysheets <- read_excel_allsheets("./2011 Lab Results.xls")
df=clean(mysheets)
df2011 <- df
df2011$year <- 2011

mysheets <- read_excel_allsheets("./2012 Lab Results.xls")
df=clean(mysheets)
df2012 <- df
df2012$year <- 2012

mysheets <- read_excel_allsheets("./2013 Lab Results.xls")
df=clean(mysheets)
df2013 <- df
df2013$year <- 2013

mysheets <- read_excel_allsheets("./2014 Lab Results.xls")
df=clean(mysheets)
df2014 <- df
df2014$year <- 2014

mysheets <- read_excel_allsheets("./2015 Lab Results.xlsx")  
df=clean(mysheets)
df2015 <- df
df2015$year <- 2015


#merge into one data frame
final <- rbind(df2006, df2007, df2008, df2009, df2010, df2011, df2012, df2013, df2014, df2015)  
names(final)
final <- final[c(9,1:8)]
names(final)[names(final) == '.id'] <- 'Date'


#Remove leading < or > in reading and e.coli columns
remember <- function (x) { #first create column to remember which values were > or <
      ifelse(grepl(">",x),"right censoring", ifelse(grepl("<",x), "left censoring", NA))  
}

final$Reading.1.Removed=remember(final$Reading.1)
final$Reading.2.Removed=remember(final$Reading.2)
final$E.coli.Removed=remember(final$Escherichia.coli)

#Remove all less than and greater than signs
from <- c('>','<')
to <- c("", "")

gsub2 <- function(pattern, replacement, x, ...) { 
      for(i in 1:length(pattern))
            x <- gsub(pattern[i], replacement[i], x, ...)
      x
}

final$Reading.1=gsub2(from, to, final$Reading.1) #this makes <1 a 1 and >2419.6 a 2419.60 
final$Reading.2=gsub2(from, to, final$Reading.2)
final$Escherichia.coli=gsub2(from, to, final$Escherichia.coli)
final[5:7]<-lapply(final[5:7], as.numeric) 


#create geometric mean of Reading.1 and Reading.2
final$e.coli.geomean=round(apply(final[,c(5,6)],1,geometric.mean, na.rm=T), 1) 
final=final[c(1:7, 13, 8:12)]
 
#TO add the e.coli values where there are NO READINGS -- note these are TRUNCATED so values may be slightly off (not rounded)
final[which(!is.na(final[7]) & is.na(final[8])),][8] <- final[which(!is.na(final[7]) & is.na(final[8])),][7] 
final=final[c(1:6,8:13)]

#create 1/0 for advisory at or over 235
final$advisory<-ifelse(final$e.coli.geomean>=235, 1, 0)


#clean the PM dates (June 16 2015; July 1 2014/2015)
PM <- function (x) {  
      ifelse(grepl("PM",x),"PM", NA)  
}

final$Date.PM.Removed=PM(final$Date) #to remember which readings were PM specific
from.1 <- c(' \\(PM\\)',' PM')
to.1 <- c("", "")
final$Date=gsub2(from.1, to.1, final$Date) #get rid of the PMs

#make Dates easier to use
final=unite_(final, "newdate", c("Date", "year"), sep=" ", remove=F)
final$Full_date <- as.Date(final$newdate, format="%B %d %Y") #now dates are sortable
final$Weekday <- weekdays(final$Full_date) #add day of week
final$Month <- format(final$Full_date,"%B")
final$Day <- format(final$Full_date, "%d")
final=final[c(16, 2, 17:19,15, 4:8, 14, 9:13)]


#Remove problematic dates
final <- final[-which(final$Full_date %in% c(as.Date("2006-07-06"), as.Date("2006-07-08"), as.Date("2006-07-09"))),]


#Clean Beach Names

#Remove outlier with 6488.0 value



write.csv(final, "lab_results.csv", row.names=FALSE)
 

