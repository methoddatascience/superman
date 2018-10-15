# Some findings

## Target Value

* Average revenue of weekends is significantly lower than that of weekdays

* A peak in number of sessions and visitors from 2016-10 to 2017-1. What's the cause? Revenue didn't go up as visits, why?

## Train/Test distributions

### channelGrouping

* Much lower ratio of social traffic in test
* Much higer ratio of display in test

### device.deviceCategory

* Lower desktop, higher mobile in test

### trafficSource.medium

* Much higher CPC in test

### 'trafficSource.adwordsClickInfo.adNetworkType'

* 'Content' does not exist in train


## Others

* 'totals.hits' max 500? Highly correlated with 'totals.pageviews', not highly correlated with revenue

## Future steps

* Interactions among ads, campaign, visitor and revnue?
* More insight into categorical variables with many unique values

