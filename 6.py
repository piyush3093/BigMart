''' LightGBM '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train data.csv')
test_data = pd.read_csv('test data.csv')

train_set = pd.read_csv('train_set.csv')
valid_set = pd.read_csv('valid_set.csv')

train_set = train_set.drop('Unnamed: 0', axis = 1)
valid_set = valid_set.drop('Unnamed: 0', axis = 1)

# Feature Creation
group = train_set.Item_Type.groupby([train_set['Outlet_Identifier'], train_set['Item_Type']]).count()
value_dict = group.to_dict()

train_set['Types of item in outlet'] = np.nan
valid_set['Types of item in outlet'] = np.nan
train_set = train_set.set_index(pd.Index([i for i in range(0, len(train_set))]))
valid_set = valid_set.set_index(pd.Index([i for i in range(0, len(valid_set))]))

for i in range(0, len(train_set)):
    train_set['Types of item in outlet'][i] = group[train_set.Outlet_Identifier[i]][train_set.Item_Type[i]]

for i in range(0, len(valid_set)):
    valid_set['Types of item in outlet'][i] = group[valid_set.Outlet_Identifier[i]][valid_set.Item_Type[i]]

data = pd.concat([train_data.iloc[:, :11], test_data], axis = 0)
values = data.Item_Identifier.value_counts().to_dict()
    
train_set['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(train_set)):
    train_set.No_of_stores_which_sell_that_thing[i] = int(values[train_set.Item_Identifier[i]])
valid_set['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(valid_set)):
    valid_set.No_of_stores_which_sell_that_thing[i] = int(values[valid_set.Item_Identifier[i]])

# One hot encoding

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

train_valid_set = pd.concat([train_set, valid_set], axis = 0)

train_valid_set, dummy = encoding(train_valid_set, 'Establishment_Period')
train_valid_set = train_valid_set.drop(['Item_Identifier', 'Item_Fat_Content', 'Outlet_Establishment_Year'], axis = 1)

train_set = train_valid_set.iloc[:6984, :]
valid_set = train_valid_set.iloc[6984:, :]

y_train = np.sqrt(train_set.loc[:, 'Item_Outlet_Sales'])
y_valid = np.sqrt(valid_set.loc[:, 'Item_Outlet_Sales'])

train_set = train_set.drop('Item_Outlet_Sales', axis = 1)
valid_set = valid_set.drop('Item_Outlet_Sales', axis = 1)

# Selecting Best Features

from sklearn.feature_selection import SelectKBest, f_regression
selectkbest = SelectKBest(score_func = f_regression, k = 23)
selectkbest.fit(train_set.iloc[:, 5:], y_train)
for i in range(len(selectkbest.scores_)):
    print(' %s: %f' % (train_set.iloc[:, 5:].columns[i], selectkbest.scores_[i]))
    
train_data_cat = selectkbest.transform(train_set.iloc[:, 5:])
valid_data_cat = selectkbest.transform(valid_set.iloc[:, 5:])

train_data_cont = train_set.iloc[:, :5].values
valid_set_cont = valid_set.iloc[:, :5].values

train_set_new = np.concatenate([train_data_cont, train_data_cat], axis = 1)
valid_set_new = np.concatenate([valid_set_cont, valid_data_cat], axis = 1)

train_set_new = np.delete(train_set_new, 3, axis = 1)
valid_set_new = np.delete(valid_set_new, 3, axis = 1)

# Model

from lightgbm import LGBMRegressor
regressor = LGBMRegressor(max_depth = 5, feature_fraction = 0.6, min_data_in_leaf = 20, learning_rate = 0.05)
regressor.fit(train_set_new, y_train)
y_pred = regressor.predict(valid_set_new)
np.sqrt(mean_squared_error(np.square(y_valid), np.square(y_pred)))

from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(np.square(y_valid), np.square(y_pred)))

# Final Model and Predictions
#%%

for i in range(0, len(train_data)):
    if train_data.Item_Fat_Content[i] == 'low fat':
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'LF'):
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'reg'):
        train_data.Item_Fat_Content[i] = 'Regular'
    else:
        continue

for i in range(0, len(test_data)):
    if test_data.Item_Fat_Content[i] == 'low fat':
        test_data.Item_Fat_Content[i] = 'Low Fat'
    elif(test_data.Item_Fat_Content[i] == 'LF'):
        test_data.Item_Fat_Content[i] = 'Low Fat'
    elif(test_data.Item_Fat_Content[i] == 'reg'):
        test_data.Item_Fat_Content[i] = 'Regular'
    else:
        continue

train_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')
test_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')

train_data.Item_Visibility = train_data.Item_Visibility.replace(0, 0.070482)
test_data.Item_Visibility = train_data.Item_Visibility.replace(0, 0.070482)

for i in range(0, len(train_data)):
    if (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Low Fat'):
        train_data.Item_Weight[i] = 12.940000
    elif (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Regular'):
        train_data.Item_Weight[i] = 12.710000

for i in range(0, len(test_data)):
    if (np.isnan(test_data.Item_Weight[i]) and test_data.Item_Fat_Content[i] == 'Low Fat'):
        test_data.Item_Weight[i] = 12.940000
    elif (np.isnan(test_data.Item_Weight[i]) and test_data.Item_Fat_Content[i] == 'Regular'):
        test_data.Item_Weight[i] = 12.710000

train_data['Establishment_Period'] = pd.cut(x = train_data['Outlet_Establishment_Year'], bins = [1984, 1991, 1997, 2003, 2009], labels = ['very old', 'old', 'medium', 'new'])
test_data['Establishment_Period'] = pd.cut(x = test_data['Outlet_Establishment_Year'], bins = [1984, 1991, 1997, 2003, 2009], labels = ['very old', 'old', 'medium', 'new'])

group = train_data.Item_Type.groupby([train_data['Outlet_Identifier'], train_data['Item_Type']]).count()
value_dict = group.to_dict()

train_data['Types of item in outlet'] = np.nan
test_data['Types of item in outlet'] = np.nan

for i in range(0, len(train_data)):
    train_data['Types of item in outlet'][i] = group[train_data.Outlet_Identifier[i]][train_data.Item_Type[i]]

for i in range(0, len(test_data)):
    test_data['Types of item in outlet'][i] = group[test_data.Outlet_Identifier[i]][test_data.Item_Type[i]]

data = pd.concat([train_data.iloc[:, :11], test_data], axis = 0)
values = data.Item_Identifier.value_counts().to_dict()
    
train_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(train_set)):
    train_data.No_of_stores_which_sell_that_thing[i] = int(values[train_data.Item_Identifier[i]])
test_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(test_data)):
    test_data.No_of_stores_which_sell_that_thing[i] = int(values[test_data.Item_Identifier[i]])

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

train_data = train_data.drop('Item_Outlet_Sales', axis = 1)
data = pd.concat([train_data, test_data], axis = 0)

data, dummy = encoding(data, 'Establishment_Period')
data = data.drop(['Item_Identifier', 'Item_Fat_Content', 'Outlet_Establishment_Year'], axis = 1)

train_data = data.iloc[:8523, :]
test_data = data.iloc[8523:, :]

from sklearn.feature_selection import SelectKBest, f_regression
selectkbest = SelectKBest(score_func = f_regression, k = 23)
selectkbest.fit(train_data.iloc[:, 5:], y_train)
for i in range(len(selectkbest.scores_)):
    print(' %s: %f' % (train_set.iloc[:, 5:].columns[i], selectkbest.scores_[i]))
    
train_data_cat = selectkbest.transform(train_data.iloc[:, 5:])
test_data_cat = selectkbest.transform(test_data.iloc[:, 5:])

train_data_cont = train_data.iloc[:, :5].values
test_data_cont = test_data.iloc[:, :5].values

X_train = np.concatenate([train_data_cont, train_data_cat], axis = 1)
X_test = np.concatenate([test_data_cont, test_data_cat], axis = 1)

from lightgbm import LGBMRegressor
regressor = LGBMRegressor(max_depth = 5, feature_fraction = 0.6, min_data_in_leaf = 20, learning_rate = 0.05)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

Solution = np.square(pd.DataFrame(y_pred))
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data_new['Item_Identifier'], test_data_new['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_7.csv', index = False)

test_data_new = pd.read_csv('test data.csv')

# XGBoost
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.model_selection import GridSearchCV
param_grid = {
        'max_depth': [3, 6, 10],
        'min_child_weight': [0.5, 1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
        }

grid = GridSearchCV(estimator = regressor, param_grid = param_grid, refit=True, verbose=2, n_jobs = -1, scoring = 'neg_root_mean_squared_error')
grid.fit(X_train ,y_train)
print(grid.best_estimator_)

Solution = np.square(pd.DataFrame(y_pred))
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data_new['Item_Identifier'], test_data_new['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_7.csv', index = False)

#%%




