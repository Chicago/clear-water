library(data.table)
library(ggplot2)

# x - latitude
# y - freq of elevated occurances

clusters <- 5
reclusters <- 5

source("R/00_Startup.R")
df <- readRDS(paste0(getwd(),"/Data/df.Rds"))
dt <- data.table(df)

dt <- dt[Year %in% c("2006",
                     "2007",
                     "2008",
                     "2009",
                     "2010",
                     "2011",
                     "2012",
                     "2013",
                     "2014",
                     "2015",
                     "2016",
                     "2017")]

dt[Escherichia.coli >= 235, exceedance := 1]
dt[Escherichia.coli < 235, exceedance := 0]
dt_byBeach <- dt[!is.na(exceedance),
                 .(exceedances = sum(exceedance)),
                 .(Client.ID, Latitude, Longitude)]
dt_byBeach$breakwater <- 0
dt_byBeach[Client.ID == "12th"]$breakwater <- 108
dt_byBeach[Client.ID == "31st"]$breakwater <- 415
dt_byBeach[Client.ID == "39th"]$breakwater <- 128
dt_byBeach[Client.ID == "57th"]$breakwater <- 442
dt_byBeach[Client.ID == "63rd"]$breakwater <- 1215
dt_byBeach[Client.ID == "Albion"]$breakwater <- 0
dt_byBeach[Client.ID == "Calumet"]$breakwater <- 536
dt_byBeach[Client.ID == "Foster"]$breakwater <- 202
dt_byBeach[Client.ID == "Howard"]$breakwater <- 0
dt_byBeach[Client.ID == "Juneway"]$breakwater <- 0
dt_byBeach[Client.ID == "Leone"]$breakwater <- 41
dt_byBeach[Client.ID == "Montrose"]$breakwater <- 977
dt_byBeach[Client.ID == "North Avenue"]$breakwater <- 499
dt_byBeach[Client.ID == "Oak Street"]$breakwater <- 0
dt_byBeach[Client.ID == "Ohio"]$breakwater <- 1381
dt_byBeach[Client.ID == "Osterman"]$breakwater <- 265
dt_byBeach[Client.ID == "Rainbow"]$breakwater <- 1002
dt_byBeach[Client.ID == "Rogers"]$breakwater <- 75
dt_byBeach[Client.ID == "South Shore"]$breakwater <- 387
dt_byBeach[Client.ID == "Jarvis"]$breakwater <- 0


set.seed(317)
km <- kmeans(scale(dt_byBeach[,2:5]),
             centers = clusters,
             nstart = 100)
plot(dt_byBeach[,3:2],
     col =(km$cluster +1),
     main=paste0("K-Means result with ", clusters, " clusters"), 
     pch=20, 
     cex=2)
dt_byBeach$cluster <- km$cluster

dt_byBeach[order(exceedances, decreasing = TRUE)]

# remove worst clusters (these are the beaches to always test)

km2 <- kmeans(scale(dt_byBeach[!cluster %in% c(2,5),
                         2:3]),
             centers = reclusters,
             nstart = 100)

plot(dt_byBeach[!cluster %in% c(2,5),
                2:3],
     col =(km2$cluster +1),
     main=paste0("K-Means result with ", reclusters, " reclusters"), 
     pch=20, 
     cex=2)
dt_byBeach[!cluster %in% c(2,5),"recluster"] <- km2$cluster

dt_byBeach[order(exceedances, decreasing = TRUE)]

lm(exceedances ~ breakwater, dt_byBeach)

cbPalette <- c("#999999", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
dt_byBeach$cluster <- as.factor(dt_byBeach$cluster)
ggplot(dt_byBeach, aes(breakwater, exceedances, color = cluster, shape = cluster)) +
  geom_point(size = 3) + 
  geom_abline(slope = .1202, intercept = 60.8528, linetype = "dashed") +
  labs(x = "Breakwater Length (ft)", y = "Total E. coli Exceedances", title = "Chicago Beaches 2006 - 2017") +
  guides(fill = "legend") +
  scale_fill_manual(values=cbPalette)

