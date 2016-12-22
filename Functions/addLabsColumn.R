#This function is used to add a beach's lab results fo the day as a column
#For example, you could add the Montrose culture geomean as a column
addLabsColumn <- function(df, beach, column) {
  new_col <- c()
  for (row in c(1:nrow(df))) {
    this_date <- df$Date[row]
    new_value <- df[df$Client.ID == beach & df$Date == this_date,column]
    if (length(new_value) == 0)
      new_col <- c(new_col, NA)
    else
      new_col <- c(new_col, new_value)
  }
  df <- cbind(df, new_col)
  colnames(df)[ncol(df)] <- paste0(beach,"_",column)
  df
}