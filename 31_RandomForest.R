#To be able to use the DNA testing (very expensive), we want to try and cluster
#beaches with similar beaches to be able to predict the levels of E.coli at
#other beaches to reduce the cost in the future on how much is spent on testing.

#In this program we split the beaches into clusters, the clusters were chosen by 
#the K-means method. Within the clusters we are going to :
#1- Use only one beach as our predicting beach and assume we know the correct
#   mean for each day, therefore we cannot use that beach in 
#2-

#Create the clusters
Calumet_Cluster<- c("31st","Calumet","South Shore")
Rainbow_Cluster<- c("Rainbow")
SixtyThird_Cluster<- c("63rd")
Montrose_Cluster<- c("Montrose")
Southern_Cluster<- c("57th","12th","39th")
Northern_Cluster<- c("Albion","Foster","Howard","Jarvis","Juneway","Leone",
                     "North Avenue", "Oak Street", "Ohio", "Osterman", "Rogers")

#Create the variables that the RF is going to run on
predictor_variables<-c("Client.ID","Day","Month","Year")
numeric_variables<-c(
  "Escherichia.coli",
  "DayOfYear",
  "precipIntensity",
  "temperatureMax",
  "temperatureMin",
  "humidity",
  "Water.Level")
#A List of all the clusters to use later in getting the beaches assigned to the 
#correct cluster.
clusters<-list(Calumet_Cluster, Rainbow_Cluster, SixtyThird_Cluster, Montrose_Cluster, Southern_Cluster, Northern_Cluster)
client =c("Calumet","Rainbow","63rd","Montrose","57th","Rogers")

predict_year<-2006
low_cutoff <-20
high_cutoff<-500
final<-NULL


for(j in 1:length(client))
{
  Cluster_df<- df[df$Client.ID%in%clusters[[j]],c(predictor_variables,numeric_variables)]
  for(i in 1:length(numeric_variables)){
    Calumet_Cluster_df$numeric_variables[i]<-as.numeric(Calumet_Cluster_df$Calumet_Cluster_df$numeric_variables[i])
  }
  rm(i)
  
  Cluster_df<- na.omit(Cluster_df)
  
  known_beach_df<-Cluster_df[Cluster_df$Client.ID==client[j],c("Day",
                                                               "Month",
                                                               "Year",
                                                               "Escherichia.coli")]
  names(known_beach_df)[names(known_beach_df)== 'Escherichia.coli']<-"Known_Beach.Escherichia.coli"
  
  Cluster_df<-merge(Cluster_df,known_beach_df)
  Cluster_df<- na.omit(Cluster_df)
  
  
  predicting_beaches<- clusters[[j]][clusters[[j]]!=client[j]]
  if(length(predicting_beaches)>0)
  {
    for(k in 1:length(predicting_beaches))
    {
      predicting_df<-Cluster_df[Cluster_df$Client.ID == predicting_beaches[k],]
      test<-subset(predicting_df,Year == as.character(predict_year))
      train_low <- subset(predicting_df,Escherichia.coli<=low_cutoff
                          & Escherichia.coli>5
                          & Client.ID != client[j] & Year != as.character(predict_year))
      train_high<-subset(predicting_df,Escherichia.coli>=high_cutoff 
                         & Client.ID != client[j] & Year != as.character(predict_year))
      training<-rbind(train_low,train_high)
      training$Escherichia.coli<- ifelse(training$Escherichia.coli<235,0,1)
      test$Escherichia.coli<- ifelse(test$Escherichia.coli<235,0,1)
      training<- training[,!(names(training)%in%predictor_variables)]
      test<- test[,!(names(test)%in%predictor_variables)]
      model<-randomForest(factor(Escherichia.coli)~.,data=training)
      test$pred<-predict(model,newdata = test)
      final<-rbind(final,test)
    }
  }
} 
table(final$Escherichia.coli,final$pred)

