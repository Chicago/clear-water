#This function is used to add a beach's lab results for the day as a column
#For example, you could add the Montrose culture geomean (for the row's day)  as a column
addLabsColumn <- function(df, beach, column) {
  new_col <- c()
  for (row in c(1:nrow(df))) {
    this_doy <- df$DayOfYear[row]
    this_year <- df$Year[row]
    new_value <- df[df$Client.ID == beach & df$DayOfYear == this_doy & df$Year == this_year,column]
    if (length(new_value) == 0)
      new_col <- c(new_col, NA)
    else
      new_col <- c(new_col, new_value)
  }
  print(paste0("Adding Column: ",beach,"_",column))
  df <- cbind(df, new_col)
  colnames(df)[ncol(df)] <- paste0(beach,"_",column)
  df
}