import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

train = load_df(csv_path='train.csv')
test = load_df(csv_path='test.csv')

print('Processing json completed!')

# Read data and convert date column to datetime
train['date'] = train['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

test['date'] = test['date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

# Drop constant columns
drop_train = ['socialEngagementType', 'device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'totals.visits', 'trafficSource.adwordsClickInfo.criteriaParameters', 'trafficSource.campaignCode']

drop_test = ['socialEngagementType', 'device.browserSize', 'device.browserVersion', 'device.flashVersion', 'device.language', 'device.mobileDeviceBranding', 'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName', 'device.mobileDeviceModel', 'device.mobileInputSelector', 'device.operatingSystemVersion', 'device.screenColors', 'device.screenResolution', 'geoNetwork.cityId', 'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.networkLocation', 'totals.visits', 'trafficSource.adwordsClickInfo.criteriaParameters']

train = train.drop(drop_train, axis=1)
test = test.drop(drop_test, axis=1)

print('drop cols complete')

# Fill some nulls with 0
train_fillzero = ['totals.bounces', 'totals.newVisits', 'totals.pageviews', 
                 'totals.transactionRevenue']
test_fillzero = ['totals.bounces', 'totals.newVisits', 'totals.pageviews']

train.loc[:,train_fillzero] = train.loc[:,train_fillzero].fillna(0.0)
test.loc[:,test_fillzero] = test.loc[:,test_fillzero].fillna(0.0)
print('Cleaning num complete')

# Fill some nulls with 'unknown'
train_cat_nulls = ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType','trafficSource.adwordsClickInfo.gclId','trafficSource.adwordsClickInfo.isVideoAd',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.keyword', 'trafficSource.referralPath',
       'trafficSource.isTrueDirect']
train.loc[:, train_cat_nulls] = train.loc[:,train_cat_nulls].fillna('unknown')
test.loc[:, train_cat_nulls] = test.loc[:,train_cat_nulls].fillna('unknown')

print('Cleaning cat complete')

# Cast
train.loc[:,test_fillzero] = train.loc[:,test_fillzero].astype('int64')
test.loc[:,test_fillzero] = test.loc[:,test_fillzero].astype('int64')
train.loc[:,'totals.transactionRevenue'] = train.loc[:,'totals.transactionRevenue'].astype('float64')
train.loc[:,'totals.hits'] = train.loc[:,'totals.hits'].astype('int64')
test.loc[:,'totals.hits'] = test.loc[:,'totals.hits'].astype('int64')
train.loc[:,'visitStartTime'] = pd.to_datetime(train.loc[:,'visitStartTime'],unit='s')
test.loc[:,'visitStartTime'] = pd.to_datetime(test.loc[:,'visitStartTime'],unit='s')


print('Train and Test ready for use')

#total = pd.concat([train, test], axis=0, sort=False)

#train.to_csv('train_cleaned.csv', index=False)
#test.to_csv('test_cleaned.csv', index=False)

