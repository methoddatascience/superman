getwd()
library(tidyr)
library(plyr)
library(dplyr)
library(lubridate)
library(jsonlite)
library(ggplot2)
library(tidyverse)

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

#Parsing the json columns
Temp2 <- function(x){
  paste("[", paste(x, collapse = ","), "]") %>% fromJSON(flatten = T)
}




T2 <- Temp2(train$device)
Geo_nwk <- Temp2(train$geoNetwork)
Totals <- Temp2(train$totals)
traffic <- Temp2(train$trafficSource)


#Combine the columns into the training dataset
train <- cbind(train,T2,Geo_nwk,Totals, traffic )

train<- subset(train, select = -c(device, geoNetwork, totals,trafficSource))


#number of transactions

train$transactionRevenue <- as.numeric(train$transactionRevenue)

table(!is.na(train$transactionRevenue))


#Which channel produced the most revenue ?
#Which Channel has the highest Average order value?

DF <- train %>% group_by(channelGrouping) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
count.n = sum(!is.na(transactionRevenue)),
AOV = sum/count.n)

DF

#Display channel has the highest AOV - Plot
ggplot(aes(x = channelGrouping, y= sum), data = DF) + geom_bar(stat = "identity") 



#How many are unique visitors and how many are returning?
table(duplicated(train$fullVisitorId))

#duplicate sessions
table(duplicated(train$sessionId))

#Creating a subset of only transactions
Transact_subset <- subset(train,is.na(transactionRevenue) == FALSE)


#People that made multiple purchases?
table(duplicated(Transact_subset$fullVisitorId))
#1551 


#Plot TransactionsRevenue vs visit number
ggplot(aes(x = visitNumber, y = transactionRevenue), data = train) + geom_point() + 
  xlim(0,150)

train$hits <- as.integer(train$hits)

train$country <-as.factor(train$country)

#Which country is producing the highest revenue?
#US,CA, Venezuela, Japan, Kenya
DF_country <- train %>% group_by(country) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue))) %>%
  ungroup()

head(DF_country)

DF_country[order(-DF_country$sum),]

#Concatenating source-medium

train  <-unite(train, sourcemedium, c(source,medium),sep= "/", remove = FALSE)
names(train)

train[-43]

#Source-medium producing revenue:

DF_sourcemedium <- train %>% group_by(sourcemedium) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue)),
            AOV = sum/count.n)

DF_sourcemedium[order(-DF_sourcemedium$sum),]


#Grouping by medium
DF_medium <- train %>% group_by(medium) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue)),
            AOV = sum/count.n)

DF_medium[order(-DF_medium$sum),]


#Keyword producing Revenue

DF_KW <- train %>% group_by(keyword) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue)),
           AOV = sum/count.n)

DF_KW[order(-DF_KW$sum),]

#Campaigns
DF_campaign <- train %>% group_by(campaign) %>% 
  summarise(sum = sum(transactionRevenue, na.rm = TRUE), 
            count.n = sum(!is.na(transactionRevenue)),
            AOV = sum/count.n)


DF_campaign[order(-DF_campaign$sum),]

#Columns to be removed
table(train$socialEngagementType)
table(train$browserVersion)
table(train$operatingSystemVersion)
table(train$mobileDeviceBranding)
table(train$mobileInputSelector)
table(train$mobileDeviceInfo)
table(train$mobileDeviceMarketingName)
table(train$flashVersion)
table(train$language)
table(train$screenColors)
table(train$screenResolution)
table(train$latitude)
table(train$longitude)
table(train$networkLocation)
table(train$adwordsClickInfo.criteriaParameters)
table(train$adwordsClickInfo.slot)
table(train$adwordsClickInfo.page)
table(train$adContent)
table(train$browserSize)

names(train)
train<- train[-c(5,10,13,15,17,18,19,20,21,22,23,33,34,35,51,53,52)]
train <- train[-9]

class(train$bounces)
train$bounces <- as.numeric(train$bounces)

class(train$pageviews)
train$pageviews <- as.numeric(train$pageviews)
train$source <- as.factor(train$source)
train$medium <- as.factor(train$medium)
train$visits <- as.numeric(train$visits)
library(rpart)
library(rpart.plot)

table(duplicated(train$visitId))

library(RcppRoll)

#Feature : Total interaction
Dataframe <- select(train, visitId,fullVisitorId, hits, pageviews)

Feature <- Dataframe%>% group_by( fullVisitorId) %>%
  mutate(TotalInteraction =cumsum(hits)*cumsum(pageviews)) 

train1 <- merge(train1 , Feature, by = 'visitId')

