
# coding: utf-8

# In[258]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[259]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize



# In[260]:


train_df = pd.read_csv('C:/Users/Imp/KAGGLE/Ganalytics/data/train-flattened.csv')


# In[261]:


test_df = pd.read_csv('C:/Users/Imp/KAGGLE/Ganalytics/data/test-flattened.csv')


# In[268]:


print(train_df.shape, test_df.shape)


# In[269]:


train_df.head()


# Firstly we have Visitor Id vs Total Revenue per Id

# In[270]:


train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
Vis_rev = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(8,6))
plt.scatter(range(Vis_rev.shape[0]), np.sort(np.log1p(Vis_rev["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# In[271]:


Vis_rev.head(3)


# In[273]:


import seaborn as sns


# In[ ]:





# In[274]:


gdf = Vis_rev


# In[275]:


nzi = pd.notnull(train_df["totals.transactionRevenue"]).sum()
nzr = (gdf["totals.transactionRevenue"]>0).sum()
print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train_df.shape[0])
print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / gdf.shape[0])


# In[276]:


print("Number of unique visitors in train set : ",train_df.fullVisitorId.nunique(), " out of rows : ",train_df.shape[0])
print("Number of unique visitors in test set : ",test_df.fullVisitorId.nunique(), " out of rows : ",test_df.shape[0])
print("Number of common visitors in train and test set : ",len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique())) ))


# In[277]:


const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ]
const_cols


# In[278]:


import plotly
plotly.__version__


# In[279]:



import plotly.plotly as py
import plotly.graph_objs as go


# In[280]:


def horizontal_bar_chart(cnt_srs, color):
    trace = go.Bar(y=cnt_srs.index[::-1], x=cnt_srs.values[::-1], showlegend=False, orientation = 'h', marker=dict(color=color))
    return trace



# In[281]:


# Device Browser
cnt_srs = train_df.groupby('device.browser')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace1 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(50, 171, 96, 0.6)')
trace2 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(50, 171, 96, 0.6)')
trace3 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(50, 171, 96, 0.6)')

# Device Category
cnt_srs = train_df.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace4 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(71, 58, 131, 0.8)')
trace5 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10), 'rgba(71, 58, 131, 0.8)')
trace6 = horizontal_bar_chart(cnt_srs["mean"].head(10), 'rgba(71, 58, 131, 0.8)')

# Operating system
cnt_srs = train_df.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['size', 'count', 'mean'])
cnt_srs.columns = ["count", "count of non-zero revenue", "mean"]
cnt_srs = cnt_srs.sort_values(by="count", ascending=False)
trace7 = horizontal_bar_chart(cnt_srs["count"].head(10), 'rgba(246, 78, 139, 0.6)')
trace8 = horizontal_bar_chart(cnt_srs["count of non-zero revenue"].head(10),'rgba(246, 78, 139, 0.6)')
trace9 = horizontal_bar_chart(cnt_srs["mean"].head(10),'rgba(246, 78, 139, 0.6)')




# In[285]:


import datetime

def scatter_plot(cnt_srs, color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color=color,
        ),
    )
    return trace

train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
cnt_srs = train_df.groupby('date')['totals.transactionRevenue'].agg(['size', 'count'])
cnt_srs.columns = ["count", "count of non-zero revenue"]
cnt_srs = cnt_srs.sort_index()
#cnt_srs.index = cnt_srs.index.astype('str')
trace1 = scatter_plot(cnt_srs["count"], 'red')
trace2 = scatter_plot(cnt_srs["count of non-zero revenue"], 'blue')

fig = tools.make_subplots(rows=2, cols=1, vertical_spacing=0.08,
                          subplot_titles=["Date - Count", "Date - Non-zero Revenue count"])
fig.append_trace(trace1, 1, 1)
fig.append_trace(trace2, 2, 1)
fig['layout'].update(height=800, width=800, paper_bgcolor='rgb(233,233,233)', title="Date Plots")
py.iplot(fig, filename='date-plots')


# In[286]:


print("Variables not in test but in train : ", set(train_df.columns).difference(set(test_df.columns)))


# In[287]:


cols_to_drop = const_cols + ['sessionId']

train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# In[288]:


# Impute 0 for missing target values
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
train_y = train_df["totals.transactionRevenue"].values
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']


# In[289]:


import sys
from sklearn import preprocessing


# In[290]:


for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[291]:


num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)


# In[292]:


# Split the train dataset into development and valid based on time 
dev_df = train_df[train_df['date']<=datetime.date(2017,5,31)]
val_df = train_df[train_df['date']>datetime.date(2017,5,31)]


# In[293]:


dev_df.shape, val_df.shape


# In[294]:


dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
val_y = np.log1p(val_df["totals.transactionRevenue"].values)



# In[295]:


dev_X = dev_df[cat_cols + num_cols] 
val_X = val_df[cat_cols + num_cols] 
test_X = test_df[cat_cols + num_cols] 


# In[297]:


dev_X = dev_X.drop('visitStartTime',axis=1)


# In[298]:


val_X =val_X.drop('visitStartTime',axis=1)


# In[299]:


test_X = test_X.drop('visitStartTime',axis=1)


# In[300]:


import lightgbm as lgb


# In[301]:


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)


# In[302]:


pred_val


# In[303]:


from sklearn import metrics


# In[304]:


metrics.mean_squared_error(val_y,pred_val)


# In[305]:



pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[306]:


fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


# In[307]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error


# In[308]:


import xgboost as xgb
from xgboost.sklearn import XGBClassifier


# In[309]:


from sklearn import metrics   #Additional scklearn functions


# In[310]:


import matplotlib.pylab as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4


# In[311]:


from sklearn.model_selection import cross_val_score


# In[312]:


from sklearn.model_selection import GridSearchCV


# In[313]:


dev_X.shape, val_X.shape, test_X.shape


# In[314]:


dev_y.shape, val_y.shape


# In[323]:


dev_X.info()


# In[324]:


removing_columns = ['totals.bounces','totals.newVisits']


# In[325]:


test_X.shape


# In[326]:


test_X = test_X.drop(removing_columns, axis=1)


# In[327]:


test_X.shape


# In[328]:


dev_X = dev_X.drop(removing_columns, axis=1)


# In[329]:


val_X = val_X.drop(removing_columns, axis =1)


# In[330]:


trainingX=dev_X
validationX=val_X
testingX=test_X


# In[331]:


(trainingX.head(5))



# In[332]:


(validationX.head(5))



# In[333]:


(testingX.head(5))


# In[334]:


import tensorflow as tf


# In[335]:


import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
# rest of the code


# In[336]:


from keras.models import Sequential
from keras.layers import Dense


# In[337]:


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[338]:


import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

seed = 7
np.random.seed(seed)


# In[339]:


from keras import optimizers
import keras


# In[340]:


trainingX.shape, validationX.shape


# In[352]:


trainingX['totals.pageviews'].isnull().sum()


# In[365]:


trainingX['totals.pageviews']=trainingX['totals.pageviews'].fillna(0)


# In[367]:


validationX['totals.pageviews']=validationX['totals.pageviews'].fillna(0)


# In[368]:


validationX['totals.pageviews'].isnull().sum()


# In[369]:


dev_y.shape, val_y.shape


# In[370]:


from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


# In[371]:


import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[372]:


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(32, input_dim=26, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model


# In[374]:


import numpy


# In[375]:



seed = 7
numpy.random.seed(seed)
# evaluate model with dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=10, batch_size=100, verbose=0)


# In[390]:


dev_y.shape, val_y.shape


# In[393]:


trainingX.shape, test_X.shape


# In[388]:


dev_y = dev_y.reshape(-1,1)


# In[389]:


val_y = val_y.reshape(-1,1)


# In[394]:



kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, trainingX, dev_y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# In[422]:


estimator.fit(trainingX, dev_y)
prediction_NN = estimator.predict(validationX)


# In[423]:


estimator


# In[415]:



prediction_NN[prediction_NN<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(prediction_NN)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[398]:


results.mean()


# In[425]:


from sklearn.preprocessing import MinMaxScaler


# In[427]:


model2 = Sequential()
model2.add(Dense(64, input_dim=26, kernel_initializer='normal', activation='relu'))
model2.add(Dense(32, kernel_initializer='normal'))
model2.add(Dense(1))
# Compile model
model2.compile(loss='mean_squared_error', optimizer='rmsprop')


# In[435]:


from sklearn.preprocessing import StandardScaler


# In[439]:


scale = StandardScaler()


# In[441]:


scaled_x = scale.fit_transform(trainingX)


# In[442]:


scaled_validation_x = scale.transform(validationX)


# In[443]:


model2.fit(scaled_x, dev_y, epochs=10, verbose=0)


# In[444]:


predictions_NN_NN = model2.predict(scaled_validation_x)


# In[445]:


metrics.mean_squared_error(val_y,predictions_NN_NN)


# In[446]:



predictions_NN_NN[predictions_NN_NN<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(predictions_NN_NN)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# In[163]:


from pandas.util.testing import assert_frame_equal


# In[221]:


import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[400]:


import xgboost
import math


# In[401]:


xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.05, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)


# In[402]:


xgb.fit(trainingX, dev_y)


# In[403]:


predictions = xgb.predict(validationX)
metrics.mean_squared_error(val_y,predictions)


# In[404]:


#xgboost mean sqe =2.95 


# In[412]:



predictions[predictions<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(predictions)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))

