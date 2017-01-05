#To be able to use the DNA testing (very expensive), we want to try and cluster
#beaches with similar beaches to be able to predict the levels of E.coli at
#other beaches to reduce the cost in the future on how much is spent on testing.

#In this program we split the beaches into clusters, the clusters were chosen by 
#the K-means method. Within the clusters we are going

#Create the clusters
Calumet_Cluster<- c("31st","Calumet","South Shore")
Rainbow_Cluster<- c("Rainbow")
SixtyThird_Cluster<- c("63rd")
Montrose_Cluster<- c("Montrose")
Southern_Cluster<- c("57th","12th","39th")
Northern_Cluster<- c("Albion","Foster","Howard","Jarvis","Juneway","Leone",
                    "North Avenue", "Oak Street", "Ohio", "Osterman", "Rogers")

#Create the variables that the RF is going to run on
predictor_variables<-c("Client.ID")
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
percentage= .6
final<-NULL
for(j in 1:length(clusters)){
  
  Calumet_Cluster_df<- df[df$Client.ID%in%clusters[[j]],c(predictor_variables,numeric_variables)]
  
  for(i in 1:length(numeric_variables)){
    Calumet_Cluster_df$numeric_variables[i]<-as.numeric(Calumet_Cluster_df$Calumet_Cluster_df$numeric_variables[i])
  }
  rm(i)
  
  
  Calumet_Cluster_df<- na.omit(Calumet_Cluster_df)
  
  possible_train_low <- subset(Calumet_Cluster_df,Escherichia.coli<=20 & Escherichia.coli>5 & Client.ID == client[j])
  
  possible_train_high<-subset(Calumet_Cluster_df,Escherichia.coli>=500 & Client.ID == client[j])
  
  train_low<- possible_train_low[sample(nrow(possible_train_low),
                            ceiling(length(possible_train_low$Escherichia.coli)*percentage)),]
  
  train_high<- possible_train_high[sample(nrow(possible_train_high),
                                        ceiling(length(possible_train_high$Escherichia.coli)*percentage)),]
  
  training<-rbind(train_low,train_high)
  training$Escherichia.coli<- ifelse(training$Escherichia.coli<235,0,1)
  
  test<-Calumet_Cluster_df[-as.numeric(rownames(training)),]
  test$Escherichia.coli<- ifelse(test$Escherichia.coli<235,0,1)
  
  model<-randomForest(factor(Escherichia.coli)~.-Client.ID,data=training)
  
  test$pred<-predict(model,newdata = test)
  
  final<-rbind(final,test)
}
table(final$Escherichia.coli,final$pred)

calculate_roc