#Chicago Park District Raw Data Cleaning
library(readxl)  

#Slow way, but with dates -- 2006 through 2015 (except 2013)
library(plyr)
clean <- function(x) {
      ifelse(ncol(as.data.frame(x[1]))>30, x <- x[2:length(x)], x <- x[1:length(x)])  #get's rid of summary or master sheet 
      df <- ldply(x, data.frame) #puts all dfs from the list into one df
      df <- df[which(!is.na(df$Client.ID)),] #get's ride of extra rows and explainer text that appears in the sheets
      names(df)
      df <- df[1:8]
}

read_excel_allsheets <- function(filename) {  
      sheets <- readxl::excel_sheets(filename)
      x <-    lapply(sheets, function(X) readxl::read_excel(filename, sheet = X))
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

mysheets <- read_excel_allsheets("./2014 Lab Results.xls")
df=clean(mysheets)
df2014 <- df
df2014$year <- 2014

#merge into one data frame
final <- rbind(df2006, df2007, df2008, df2009, df2010, df2011, df2012, df2014) #2013, 2015 not working
names(final)
final <- final[c(9,1:8)]
names(final)[names(final) == '.id'] <- 'Date'
final[5:7]<-lapply(final[5:7], as.numeric)


write.csv(final, "lab_results.csv", row.names=FALSE)
