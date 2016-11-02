#This is to grab all of the sources and packages needed to run all these scripts.
cwd<-getwd()
file<-paste(cwd,"/Functions/usePackage.R",sep="")
source(file)
usePackage("lubridate")
usePackage("RSocrata")
