setwd("R:/MDS/Kaggle Competition")
train_GA <- read.csv("train.csv")

#Parse the json columns
json_func <- function(x){
  paste("[", paste(x, collapse = ","), "]") %>% fromJSON(flatten = T)
}

#Parse the json columns
Device <- json_func(train_GA$device)
Geo_network <- json_func(train_GA$geoNetwork)
Total <- json_func(train_GA$totals)
Traffic <- json_func(train_GA$trafficSource)

#combine columns and Removing the  original json columns 
train_GA <- cbind(train_GA,Device,Geo_network,Total,traffic)
train_GA <- subset(train_GA, select = -c(device, geoNetwork, totals,trafficSource))

#Removing columns with 1 value in the columns

train_GA <- subset( train_GA, select = -c(socialEngagementType,browserVersion,operatingSystemVersion,mobileDeviceBranding,
                                          mobileInputSelector,mobileDeviceInfo,mobileDeviceMarketingName,flashVersion,language
                                          ,screenColors,screenResolution,latitude,longitude,networkLocation,adwordsClickInfo.criteriaParameters,adwordsClickInfo.slot,
                                          adwordsClickInfo.page,adContent,browserSize, mobileDeviceModel))
#Converting class of variables

train_GA$transactionRevenue <-as.numeric( train_GA$transactionRevenue)
train_GA$hits <- as.numeric(train_GA$hits)
train_GA$pageviews <- as.numeric(train_GA$pageviews)

#FeatureEngg

#Concatenate the source and medium columns

train_GA  <- unite(train_GA, sourcemedium, c(source,medium),sep= "/", remove = TRUE)

# Feature: cumulative hits

train_GA <- train_GA %>%
  group_by(fullVisitorId)%>%
  arrange(visitStartTime) %>%
  mutate(Totalhits = cumsum(hits))

#Rows with transaction revenue, rows without transaction
train_GA_transact <- subset(train_GA, is.na(transactionRevenue)==FALSE)
train_GA_transactout <- subset(train_GA, is.na(transactionRevenue)==TRUE)

#Correlation of target with Totalhits = 0.315
cor(train_GA_transact$transactionRevenue, train_GA_transact$Totalhits)
cor(train_GA_transact$transactionRevenue, train_GA_transact$hits)

Display <- subset(train_GA_transact, channelGrouping == "Display")
  cor(Display$transactionRevenue, Display$Totalhits)

Direct <- subset(train_GA_transact, channelGrouping == "Direct")  
cor(Direct$transactionRevenue, Direct$Totalhits)

#Combining unique values
#1. imputing 0 transaction rows with 0
train_GA$transactionRevenue[is.na(train_GA$transactionRevenue)] <- 0

#create a dataframe for browser
Dataframe_browser <- train_GA %>% group_by(browser) %>% 
  summarise(
    sum.transaction = sum(transactionRevenue),
    sum.page = sum(pageviews, na.rm = TRUE),
    sum.hits = sum(hits, na.rm = TRUE),
    count.session = sum(!is.na(sessionId)),
    count.visitors = sum(!is.na(fullVisitorId)))

#keep categories only with 0 transacti0ns 
Dataframe_browser <- subset(Dataframe_browser, sum.transaction == 0)

Dataframe_browser$browserCat <- Dataframe_browser$browser
Dataframe_browser <- Dataframe_browser[,-2]


Dataframe_browser <- mutate(Dataframe_browser, 
                             visitorpercent = round((count.visitors/ sum(count.visitors)*100),2))
attach(Dataframe_browser)
Dataframe_browser$browserCat[visitorpercent < 1] = "less than 1"
Dataframe_browser$browserCat[Dataframe_browser$visitorpercent > 1 & Dataframe_browser$visitorpercent < 5 ] = "1 to 5"
Dataframe_browser$browserCat[Dataframe_browser$visitorpercent > 5 & Dataframe_browser$visitorpercent <20 ] = "5 to 20"
Dataframe_browser$browserCat[Dataframe_browser$visitorpercent > 20] = "greater than 20"

train_GA2 <- train_GA
train_GA2 <- merge(train_GA2, Dataframe_browser, by = "browser", all = TRUE)

table(is.na(train_GA2$browserCat))

#Replace nas with device browser 
train_GA2 <- transform(train_GA2, browserCat = ifelse(!is.na(browserCat), browserCat, browser))


write.csv(train_GA2,"train_browser.csv")

length(unique(train_GA2$browser))

class(train$browser)
