getwd()
library(tidyr)
library(plyr)
library(dplyr)
library(lubridate)
library(jsonlite)

setwd("R:/MDS/Kaggle Competition")
train <- read.csv("train.csv")
head(train)

dim(train)
summary(train)

train$date <- ymd(train$date)
head(train$date)

train$year <- year(train$date)
train$month <- month(train$date)
train$day <- day(train$date)


Temp2 <- function(x){
  paste("[", paste(x, collapse = ","), "]") %>% fromJSON(flatten = T)
}


#http://opensource.adobe.com/Spry/samples/data_region/JSONDataSetSample.html
#Json structure : Example 2
#https://cran.r-project.org/web/packages/jsonlite/vignettes/json-aaquickstart.html

dim(train)

T2 <- Temp2(train$device)
Geo_nwk <- Temp2(train$geoNetwork)
Totals <- Temp2(train$totals)
traffic <- Temp2(train$trafficSource)


train <- cbind(train,T2,Geo_nwk,Totals, traffic )

train<- subset(train, select = -c(device, geoNetwork, totals,trafficSource))


train$transactionRevenue <- as.numeric(train$transactionRevenue)

summary(train$transactionRevenue)

table(!is.na(train$transactionRevenue))


library(ggplot2)

#Which channel produced the most revenue ?
#Which Channel has the highest Average order value?

DF <- train %>% group_by(channelGrouping) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
count.n = sum(!is.na(transactionRevenue)),
AOV = sum/count.n)

DF

#Display channel has the highest AOV

ggplot(aes(x = channelGrouping, y= sum), data = DF) + geom_bar(stat = "identity") 
  

#How many are unique visitors and how many are returning?
table(duplicated(train$fullVisitorId))


#Creating a subset of only transactions
Transact_subset <- subset(train,is.na(transactionRevenue) == FALSE)


#People that made multiple purchases?
table(duplicated(Transact_subset$fullVisitorId))
#1551 

plot(train$visitNumber, train$transactionRevenue)

#Plot TransactionsRevenue vs visit number
ggplot(aes(x = visitNumber, y = transactionRevenue), data = train) + geom_point() + 
  xlim(0,150)

train$hits <- as.integer(train$hits)

train$country <-as.factor(train$country)

#Which country is producing the highest revenue coming from?
#US,CA, Venezuela, Japan, Kenya
DF_country <- train %>% group_by(country) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue))) %>%
  ungroup()

head(DF_country)

DF_country[order(-DF_country$sum),]





  
