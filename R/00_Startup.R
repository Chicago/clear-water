#This is to grab all of the sources and packages needed to run all these scripts.

#A function to go in and grab all the .R scripts in a file
sourceDir <- function(path, trace = TRUE, ...) {
  for (nm in list.files(path, pattern = "\\.[Rr]$")) {
    if(trace) cat(nm,":")           
    source(file.path(path, nm), ...)
    if(trace) cat("\n")
  }
}

sourceDir(paste(getwd(),"/Functions",sep=""))
usePackage("dplyr")
usePackage("lubridate")
usePackage("ggplot2")
usePackage("RSocrata")
usePackage("tidyr")
usePackage("stats")
usePackage("randomForest")

