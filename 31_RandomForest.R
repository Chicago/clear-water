# DNA testing is very expensive, to try and reduce costs we want to cluster
# beaches together to be able to predict the levels of E.coli to at the beaches 
# in Chicago. The clusters used in the script were found using a basic k-means
# clustering technique. The goal of script 31_RandomForest is to find the
# true positive, false negative, and precision rates of random forest
# predictions for the years 2006-2016 using the culture sample procedures.
# By doing this it will be used as a proof of concept for  30_Model, the actual
# model we are using for the application. Since we have a limited number of 
# ecoli days that are 'high' comparatively with the the 'low' days, we used the 
# down sampling technique. Down sampling is used to help try and distingish with
# better clarity between the 'low' and 'high' days the difference in the 
# numerical variables.

# ------------------------------------------------------------------------------
# The outline of the code:

# Picking which beaches are in the clusters, and the beaches that the DNA tests
# will be run with.

# Numerical varibles that are going to be used for the random forest model are
# chosen.

# Chose the cutoffs for downsampling

# Make a data frame with all the observations of the beaches from a cluster.

# Using the beach that the testing is done at, we will know the day of what the
# actual levels of E.coli in the water are, so we can add that to the data frame
# in a new column, `Known_Beach.Escherichia.coli`

# Make a data frame for with all the beaches in a cluster for the predicting
# year. To be used as the `test` data frame.

# Using rest of the years, and the low and high cutoffs from earlier, we create
# a we make a data frame of `training` data.

# Run a Random Forest model on each beach.

# Run a confusion Matrix at the end to get the needed values to show the
# critical numbers that come from the Random Forest model.

# ------------------------------------------------------------------------------

# tp, fn, and prec, are the critical values to be looked at in the end.
tp <- NULL
fn <- NULL
prec <- NULL
# The variable s is used at the end like in this piece of code:
# tp[s] <- conf_matrix(final$Escherichia.coli, final$pred, show = FALSE)$tpr
s <- 1
for (year in 2012:2012)
{
  # Simulating a year 20 times gives almost all the possibilities that could 
  # have happened throughout the year for the True Positive, False Negative,and 
  # Precision rates.
  for (n in 1:20)
  {
    ############################################################################
    # CLUSTER CREATION
    ############################################################################
    
    # Create the 6 clusters of beaches that are going to be used for predicting
    Calumet_Cluster <- c("31st", "Calumet", "South Shore")
    Rainbow_Cluster <- c("Rainbow")
    SixtyThird_Cluster <- c("63rd")
    Montrose_Cluster <- c("Montrose")
    Southern_Cluster <- c("57th", "12th", "39th")
    Northern_Cluster <-
      c(
        "Albion",
        "Foster",
        "Howard",
        "Jarvis",
        "Juneway",
        "Leone",
        "North Avenue",
        "Oak Street",
        "Ohio",
        "Osterman",
        "Rogers"
      )
    
    # A List of all the clusters to use later for the amount of beache in the
    # cluster that is being predicted.
    clusters <-
      list(
        Calumet_Cluster,
        Rainbow_Cluster,
        SixtyThird_Cluster,
        Montrose_Cluster,
        Southern_Cluster,
        Northern_Cluster
      )
    
    # Using the order of the clusters above in, a list of the beaches that the
    # DNA tests will be collected at is created.
    
    client = c("South Shore",
               "Rainbow",
               "63rd",
               "Montrose",
               "57th",
               "Rogers")
    
    ############################################################################
    # VARIABLE CREATION
    ############################################################################
    
    # Because the RF runs on factors and numeric variables only the
    # qualitative_variables needed to join the data frames together later,
    # but will be taken out of final data frame before the RF is run.
    # These variables should not be changed!
    qualitative_variables <- c("Client.ID", "Day", "Month", "Year")
    
    # The numeric_variables are the variables that RF is using to form it's
    # predictions, these varibles can be changed when needed.
    numeric_variables <- c("Escherichia.coli",
                           "DayOfYear",
                           "precipIntensity",
                           #  "temperatureMax",
                           #  "temperatureMin",
                           "humidity")
    #  "Water.Level")
    
    
    # The year that we are looking at, this comes from the for loop above.
    predict_year <- year
    
    # For downsampling what observations are we going to take in
    # from previous years
    low_cutoff <- 5
    high_cutoff <- 10
    
    # What is our cutoff for high E.Coli levels going to be?
    high_ecoli_level <- 150
    
    # Create a df that we are going to use as the final df to create a conf_matrix
    final <- NULL
    
    ############################################################################
    # DATA FRAME CREATION
    ############################################################################
    
    # In this for loop we are going to take each cluster and make predictions
    # within the cluster
    for (j in 1:length(client))
    {
      # Get the the qualitative_variables of the year that we are looking at
      # from the main df, and assign them to a new df called `Cluster_df`
      Cluster_df <-
        df[df$Client.ID %in% clusters[[j]], c(qualitative_variables, numeric_variables)]
      
      # Same thing as the qualitative but with the numeric variables,
      # the difference is that we change these all to numeric.
      for (i in 1:length(numeric_variables)) {
        Cluster_df$numeric_variables[i] <-
          as.numeric(Cluster_df$numeric_variables[i])
      }
      rm(i)
      
      # Take out all the rows that have an N/A in them
      Cluster_df <- na.omit(Cluster_df)
      
      # Since we are going to know the levels at a particular beach, Use those
      # levels to try and predict at other beaches in the cluster.
      known_beach_df <-
        Cluster_df[Cluster_df$Client.ID == client[j], c("Day",
                                                        "Month",
                                                        "Year",
                                                        "Escherichia.coli")]
      # Change the column name from `Escherichia.coli` to
      # `Known_Beach.Escherichia.coli` so we don't have 2 `Escherichia.coli` 
      # when the merge happens.
      names(known_beach_df)[names(known_beach_df) == 'Escherichia.coli'] <-
        "Known_Beach.Escherichia.coli"
      
      # Merge cluster_df and known_beach_df and check for any N/A's
      Cluster_df <- merge(Cluster_df, known_beach_df)
      Cluster_df <- na.omit(Cluster_df)
      
      # Create a variable that has only the beaches we are predicting, so we can
      # reduce the data frame in the future.
      predicting_beaches <- clusters[[j]][clusters[[j]] != client[j]]
      
      
      ##########################################################################
      # MODEL CREATION
      ##########################################################################
      
      # If we are predicting for 1 or more beaches, we will do all the
      # predictions for the specific beach
      if (length(predicting_beaches) > 0)
      {
        for (k in 1:length(predicting_beaches))
        {
          # Build a data frame for the beach we are predicting
          predicting_df <-
            Cluster_df[Cluster_df$Client.ID == predicting_beaches[k], ]
          
          # Build the testing and training sets
          test <-
            subset(predicting_df, Year == as.character(predict_year))
          train_low <-
            subset(
              predicting_df,
              Escherichia.coli <= low_cutoff
              & Escherichia.coli > 5
              &
                Client.ID != client[j] & Year != as.character(predict_year)
            )
          train_high <-
            subset(
              predicting_df,
              Escherichia.coli >= high_cutoff
              &
                Client.ID != client[j] & Year != as.character(predict_year)
            )
          # Put the low and high training sets together
          training <- rbind(train_low, train_high)
          
          # Set the binary outcome for the test and training sets
          training$Escherichia.coli <-
            ifelse(training$Escherichia.coli < high_ecoli_level, 0, 1)
          test$Escherichia.coli <-
            ifelse(test$Escherichia.coli < high_ecoli_level, 0, 1)
          
          # Take the Qualitative variables out of the data frame
          training <-
            training[, !(names(training) %in% qualitative_variables)]
          test <- test[, !(names(test) %in% qualitative_variables)]
          
          # Run the RandomForest model
          model <-
            randomForest(factor(Escherichia.coli) ~ ., data = training)
          # Make Predictions on the test set
          test$pred <- predict(model, newdata = test)
          # Put all the predictions from all beaches together into 1 data set
          final <- rbind(final, test)
        }
      }
    }
    
    # Get the necessary information that we need for ascessing the model
    tp[s] <-
      conf_matrix(final$Escherichia.coli, final$pred, show = FALSE)$tpr
    fn[s] <-
      conf_matrix(final$Escherichia.coli, final$pred, show = FALSE)$fnr
    prec[s] <-
      conf_matrix(final$Escherichia.coli, final$pred, show = FALSE)$prec
    s <- s + 1
  }
}


rm(
  Cluster_df,
  final,
  known_beach_df,
  predicting_df,
  test,
  train_high,
  train_low,
  training,
  Calumet_Cluster,
  client,
  clusters,
  high_cutoff,
  high_ecoli_level,
  j,
  k,
  low_cutoff,
  model,
  Montrose_Cluster,
  Northern_Cluster,
  numeric_variables,
  predict_year,
  qualitative_variables,
  predicting_beaches,
  Rainbow_Cluster,
  SixtyThird_Cluster,
  Southern_Cluster,
  s,
  n,
  year
)
