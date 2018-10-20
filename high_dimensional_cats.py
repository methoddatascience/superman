train.loc[:,'visitStartTime'] = pd.to_datetime(train.loc[:,'visitStartTime'])
test.loc[:,'visitStartTime'] = pd.to_datetime(test.loc[:,'visitStartTime'])

train.loc[:,'date'] = pd.to_datetime(train.loc[:,'date'])
test.loc[:,'date'] = pd.to_datetime(test.loc[:,'date'])

# Combine metro and city as an identifier for location
train['metro-city'] = train['geoNetwork.metro'] + '-' + train['geoNetwork.city']
test['metro-city'] = test['geoNetwork.metro'] + '-' + test['geoNetwork.city']

total = pd.concat([train, test], axis=0, sort=False)

# Variables with large number of unique values
cats = ['channelGrouping','device.browser', 'device.operatingSystem',
       'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.page', 'trafficSource.campaign', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source', 'metro-city']

# Return corresponding average revenue of each value in a variable
def cat_sess_rev(colname):
    df = train[[colname, 'totals.transactionRevenue']].groupby([colname]).mean().reset_index()
    df.columns = [colname, 'mean_revenue']
    df = df.sort_values(by='mean_revenue', ascending=False)
    return df

# Get a set of values in a variable that generate revenue
def non_zero_rev(colname):
    df = cat_sess_rev(colname)
    with_rev = set(df[df['mean_revenue']>0][colname])
    return with_rev

# Recode values that don't have revenue
def recode_zero_rev(val, with_rev):
    if val not in with_rev:
        return 'noRev'
    else:
        return val
    
# Recode each categorical variables
for i in cats:
    with_rev_set = non_zero_rev(i)
    total.loc[:,i] = total[i].apply(lambda x: recode_zero_rev(x, with_rev_set))
    print(i+' recoding complete')
    