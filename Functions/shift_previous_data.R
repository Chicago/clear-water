#This function can be called with or without the vector of column names, names_of_columns_to_shift
#without the argument names_of_columns_to_shift, it defaults to all numeric columns in original_data_frame
shift_previous_data <- function(number_of_observations, original_data_frame, names_of_columns_to_shift=FALSE) {
  merged_data_frame <- data.frame()
  #remove rows where Client.ID == NA
  clean_data_frame <- original_data_frame[!is.na(original_data_frame$Client.ID),]         
  #subset by year
  for (year in unique(clean_data_frame$Year)) {        
    readings_by_year <- clean_data_frame[clean_data_frame$Year == year,]
    #subset by beach within year
    for (beach in unique(readings_by_year$Client.ID) ) {     
      readings_by_beach <- readings_by_year[readings_by_year$Client.ID == beach,]
      readings_by_beach <- readings_by_beach[order(readings_by_beach$Date),]
      #if no columns provided, use default (all numeric columns)
      if (names_of_columns_to_shift[1] == FALSE) {        
        readings_by_beach_columns <- readings_by_beach[sapply(readings_by_beach, is.numeric)]
        names_of_columns_to_shift <- colnames(readings_by_beach_columns)
      }
      #build new column names
      for (column in names_of_columns_to_shift) {     
        new_column_name <- paste(column,number_of_observations,"daysPrior",sep=".")
        new_column_values <- vector()
        #build new columns
        #for first n rows, use NA bc no prior data to use
        for (n in 1:number_of_observations) {         
          new_column_values[n] <- NA            
        }
        #add previous data to new columns
        for (i in number_of_observations:nrow(readings_by_beach)) {  
          new_column_values <- c(new_column_values, readings_by_beach[,column][i-n])
        }
        #merge new columns with subsetted year/beach data
        readings_by_beach <- cbind(readings_by_beach, new_column_values)    
        colnames(readings_by_beach)[colnames(readings_by_beach)=="new_column_values"] <- new_column_name
      }
      #rebuild original dataframe adding the merged data from above
      merged_data_frame <- rbind(merged_data_frame, readings_by_beach)  
    }
  }
  merged_data_frame
}