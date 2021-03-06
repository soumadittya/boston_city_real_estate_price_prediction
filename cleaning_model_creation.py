#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


housing = pd.read_csv('data.csv')


# In[3]:


# manual splitting of data into training and test set
# def split(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data)) # used for shuffling the data
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]


# In[4]:


#train_set, test_set = split(housing, 0.2)


# In[5]:


# simple splitting of data into train and test set usning sklearn
# from sklearn.model_selection import train_test_split   
# train_set, test_set = train_test_split(housing, test_size = 0.2, random_state = 42)


# In[6]:


# removing all unnamed colouns in housing dataframe
housing = housing.dropna(how='all', axis='columns')


# In[7]:


# stratified shuffle splitt using sklearn 
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[8]:


# strat_train_set['CHAS'].value_counts()


# In[9]:


# strat_test_set['CHAS'].value_counts()


# In[10]:


# 95/7


# In[11]:


# 376/28


# In[12]:


# finding standard correlation coefficients
corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[13]:


# plotting
# from pandas.plotting import scatter_matrix
# attributes = ['MEDV', 'RM', 'ZN', 'LSTAT']
# scatter_matrix(housing[attributes], figsize = (20,15), alpha = 0.8)


# In[14]:


# removing all unnamed colouns in housing dataframe
housing = housing.dropna(how='all', axis='columns')


# In[15]:


# to take care of missing attributes, we have three options:
#     1. Get rid of the misiing data points.
#     2. Get rid of the whole attribute.
#     3. Set the value to some value.


# In[16]:


# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy = 'median')
# imputer.fit(strat_train_set)


# In[17]:


# generating the missing values using imputer class of sklearn
# imputer.statistics_


# In[18]:


# assigning the data generated by imputer class of sklearn to a new variable
# imputed_data = imputer.transform(strat_train_set)


# In[19]:


# housing_tr = pd.DataFrame(imputed_data, columns = housing.columns)


# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'median')),
    ('std_scaler', StandardScaler()),
])


# In[21]:


housing_tr_numpy = my_pipeline.fit_transform(strat_train_set.iloc[:, :13])


# In[22]:


# converting numpy array housing_tr_numpy to pandas dataframe housing_tr
housing_tr = pd.DataFrame(housing_tr_numpy, columns = strat_train_set.iloc[:, :13].columns)


# In[23]:


from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
# model.fit(housing_tr_numpy[:, :13] , housing_tr_numpy[:, 13:])
model.fit(housing_tr, strat_train_set.iloc[:, 13:])


# In[77]:


sample_test = strat_test_set.iloc[:, :]


# In[78]:


sample_test_numpy = my_pipeline.transform(sample_test.iloc[:, :13])


# In[79]:


sample_test_df = pd.DataFrame(sample_test_numpy, columns = sample_test.iloc[:, :13].columns)


# In[94]:


strat_test_set


# In[85]:


model.predict(sample_test_df.iloc[:, :])


# In[28]:


sample_test


# In[49]:


# evaluationg the model
from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_tr)
mse = mean_squared_error(strat_train_set.iloc[:, 13:], housing_predictions)
rmse = np.sqrt(mse)


# In[51]:


rmse


# In[52]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_tr_numpy, strat_train_set.iloc[:, 13:], scoring = "neg_mean_squared_error")
rmse_score = np.sqrt(-scores)


# In[34]:


rmse_score


# In[55]:


def print_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std)
print_scores(rmse_score)


# In[56]:


from joblib import dump, load
dump(model, 'financial_analysis_boston.joblib') 


# #Final Testing

# In[61]:


final_test_features = strat_test_set.iloc[:, :13]
final_test_labels = strat_test_set.iloc[:, 13:]
final_test_pipeline = my_pipeline.transform(final_test_features)
final_predictions = model.predict(final_test_pipeline)
final_test_mse = mean_squared_error(final_test_labels, final_predictions)
final_rmse = np.sqrt(final_test_mse)


# In[69]:


final_rmse


# In[67]:


strat_test_set.iloc[:, 13:]


# In[72]:


strat_test_set

