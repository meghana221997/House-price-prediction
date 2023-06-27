#!/usr/bin/env python
# coding: utf-8

# <b>
# 1. Akhil Amol Bannur (ab2655)
# 2. Dikshita Baldevbhai Kashodriya (dk559)
# 3. Meghana Kunthur prakasha (mk2249)
# 4. Vikram Anegunda Umesh (va364)
# </b>

# # <b>Importing required Libraries</b>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # <b>Loading the data</b>

# In[2]:


df = pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')


# In[3]:


df.shape


# In[4]:


test_df.shape


# In[5]:


ntrain = df.shape[0]
ntest = test_df.shape[0]
val=df.drop(['SalePrice'],axis=1)
y_train = df.SalePrice
all_data = pd.concat([val, test_df])
all_data.drop(['Id'], axis=1,inplace=True)


# # <b>Removing Outliers in Training and Test data</b>

# In[6]:


num_cols = all_data.select_dtypes(include=np.number)
mean = num_cols.mean()
std = num_cols.std()
df = all_data[~((num_cols - mean).abs() > 3 * std).any(axis=1)]


# <h3>Merging both training and Testing data inorder to preprocess the data

# In[7]:


sns.heatmap(df.isnull(), cmap= 'Greens' ,yticklabels=False,cbar=False)


# <b>PRE-PROCESSING DATA</b>

# In[8]:


feature = np.array(all_data.columns.to_list())
null_val = feature[np.array(all_data.isnull().sum())>100]
all_data.shape


# In[9]:


all_data.drop(null_val, axis=1,inplace=True)
all_data.shape


# In[10]:


categorical_all_features = all_data.select_dtypes(include=['object']).columns
numerical_all_features = all_data.select_dtypes(exclude=['object']).columns


# In[11]:


for feature in categorical_all_features:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode())
    
for feature in numerical_all_features:
     all_data[feature] = all_data[feature].fillna(all_data[feature].mean())


# In[12]:


sns.heatmap(all_data.isnull(),yticklabels=False,cbar=False)


# In[13]:


def draw_correlations(df):
    correlations = df.corr()
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.heatmap(correlations)


# In[14]:


draw_correlations(all_data)


# <h3> Removing Features which have less correalation with target

# In[15]:


all_data = all_data.drop(
    ['EnclosedPorch',
     '3SsnPorch',
     'ScreenPorch',
     'PoolArea',
     'MiscVal',
     'MoSold',
     'YrSold',
     'PoolArea',
     'KitchenAbvGr',
     'BsmtHalfBath'
    ],axis=1)


# <h3> Converting Categorical data to Numerical data

# In[16]:


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

for col in categorical_all_features:
    all_data[col] = encoder.fit_transform(all_data[col])
all_data
all_data.fillna(0, inplace=True)


# In[17]:


clean_train_data = all_data[:ntrain]
clean_test_data = all_data[ntest:]


# In[18]:


corr_matrix = clean_train_data.corrwith(y_train)
corr_df = pd.DataFrame({'corr':corr_matrix})

plt.figure(figsize=(36, 36), dpi = 480)
sns.heatmap(clean_train_data[corr_df.index].corr(), annot=True, cmap='coolwarm', vmin=0, vmax=1,
            annot_kws={"size": 12})
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(clean_train_data, y_train, test_size=0.2, random_state = 1)


# In[20]:


y_train = y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)
y_train = np.log(y_train)
y_test = np.log(y_test)


# <h3> Using MinMaxScaler to transform features

# In[21]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# <h3> XGBRegressor Model

# In[22]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
xgb = XGBRegressor(learning_rate=0.09, 
                   n_estimators=1500,
                   max_depth=10,
                   subsample=0.7)
model_xgb = xgb.fit(X_train_scaled, y_train)     
score_xgb = model_xgb.score(X_test_scaled, y_test)
y_pred = model_xgb.predict(X_test_scaled)
print("XGBRegressor Model Accuracy:",score_xgb*100)
rmse = float(format(np.sqrt(mean_squared_error(y_test.astype('float64'),y_pred))))
print("\nRMSE: ", rmse)
print('MAE:', mean_absolute_error(y_test.astype('float64'), y_pred))


# <h3>Using Cross Validation score along with XGBRegressor Model

# In[23]:


import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
model = xgb.XGBRegressor()
kfold = KFold(n_splits=10, shuffle=True, random_state=200)
scores = cross_val_score(model, X_train_scaled, y_train, scoring='neg_root_mean_squared_error', cv=kfold)
print("RMSE scores:", -scores)
print("Mean RMSE score:", -scores.mean())


# <h3>Bagging regressor model using decision tree regressors

# In[24]:


from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()
bagging = BaggingRegressor(dt, n_estimators=10)
bagging.fit(X_train_scaled, y_train)
y_pred_bagging = bagging.predict(X_test_scaled)

score = bagging.score(X_test_scaled, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_bagging))
print("Bagging regressor model Accuracy:", score*100)
print("RMSE:", rmse)


# <h3> AdaBoost Regressor model using decision tree regressors

# In[25]:


from sklearn.ensemble import AdaBoostRegressor

dt = DecisionTreeRegressor()
adaboost = AdaBoostRegressor(dt, n_estimators=50)
adaboost.fit(X_train_scaled, y_train)
y_pred_boosting = adaboost.predict(X_test_scaled)

score = adaboost.score(X_test_scaled, y_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_boosting))
print("AdaBoost Regressor model Accuracy:", score*100)
print("RMSE:", rmse)


# <h3> Saving the predicted values in the sample_submission.csv file

# In[26]:


y_pred_test = model_xgb.predict(clean_test_data)
y_pred = y_pred_test.reshape(-1,1)
y_pred = np.exp(y_pred)


# In[27]:


pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],pred],axis=1)
datasets.columns=['ID','Sale Price']
datasets

