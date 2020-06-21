# Exploratory Data Analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train data.csv')
test_data = pd.read_csv('test data.csv')

sns.kdeplot(data = train_data.Item_MRP)
sns.kdeplot(data = test_data.Item_MRP)

train_data.isnull().sum()
test_data.isnull().sum()

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

train_data.Outlet_Size.value_counts()
test_data.Outlet_Size.value_counts()

group = train_data.Outlet_Size.groupby([train_data.Outlet_Type, train_data.Outlet_Size]).count()

train_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')
test_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')

train_data.Item_Visibility = train_data.Item_Visibility.replace(0, 0.070482)
test_data.Item_Visibility = test_data.Item_Visibility.replace(0, 0.070482)

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

train_data.Item_Visibility = np.sqrt(train_data.Item_Visibility)
test_data.Item_Visibility = np.sqrt(test_data.Item_Visibility)

# Creating a Validation Split

value_counts = pd.DataFrame(train_data.Item_Identifier.value_counts())
value_counts['20%'] = (value_counts.Item_Identifier * 0.2)
value_counts = value_counts.astype('int64')

item_identifier = train_data.Item_Identifier
item_identifier = item_identifier.sort_values()
item_identifier_new = item_identifier.drop_duplicates()
item_identifier_new = pd.DataFrame(item_identifier_new)
item_identifier_new['indexes'] = item_identifier_new.index
item_identifier_new = item_identifier_new.set_index(pd.Index([i for i in range(0, len(item_identifier_new))]))
item_identifier = pd.DataFrame(item_identifier)
item_identifier['indexes'] = item_identifier.index
item_identifier = item_identifier.set_index(pd.Index([i for i in range(0, len(item_identifier))]))

int(item_identifier.indexes[item_identifier[item_identifier.indexes == 7811].index + 1].values)

idx = []
for i in range(0, len(item_identifier_new)):
    item = item_identifier_new['Item_Identifier'][i]
    valid_count = value_counts['20%'][item]
    index_taken = item_identifier_new.indexes[i]
    if valid_count == 2:
        idx.append(index_taken)
        idx.append(int(item_identifier.indexes[item_identifier[item_identifier.indexes == index_taken].index + 1].values))
    if valid_count == 1:
        idx.append(index_taken)

train_sample = train_data.iloc[]

for i in range(0, len(value_counts)):
    if value_counts['Item_Identifier'][i] == 3:
        value_counts['20%'][i] = 0

for i in range(0, len(value_counts)):
    value_counts['20%'][i] = round(value_counts['20%'][i])

value_counts = value_counts.astype('int64')
indexes = pd.DataFrame(idx)
indexes.astype('int64')

valid_set = train_data.iloc[idx, :]
train_idx = [item for item in train_data.index if item not in idx]
train_set = train_data.iloc[train_idx, :]

train_set.to_csv('train_set.csv')
valid_set.to_csv('valid_set.csv')

# Testing various features

anova_group = train_data[['Establishment_Period', 'Item_Outlet_Sales']]

grps = pd.unique(anova_group.Establishment_Period.values)
d_data = {grp : anova_group['Item_Outlet_Sales'][anova_group.Establishment_Period == grp] for grp in grps}

d_data['Baking Goods']
from scipy import stats
F, p = stats.f_oneway(d_data['very old'], d_data['old'], d_data['medium'], d_data['new'])

train_set = train_set.drop(['Item_Identifier', 'Item_Fat_Content'], axis = 1)
valid_set = valid_set.drop(['Item_Identifier', 'Item_Fat_Content'], axis = 1)

# Z-Test

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train_cont_sc = min_max_x.fit_transform(train_data.iloc[:, [1, 3, 5, 13, 14]])
y_train_sc = min_max_y.fit_transform(train_data.iloc[:, 11].values.reshape(-1, 1))

from statsmodels.stats import weightstats as stests
ztest ,pval1 = stests.ztest(X_train_cont_sc[:, 0], x2=y_train_sc[:, 0], value=0,alternative='two-sided')
print(float(pval1))

# New Features

group = data.groupby([data.Item_Identifier, data.Item_Type], as_index = False).first()
group.Item_Identifier.nunique()
values = group.Item_Type.value_counts().to_dict()

data = pd.concat([train_data.iloc[:, :11], test_data], axis = 0)
values = data.Item_Identifier.value_counts().to_dict()

train_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(train_data)):
    train_data.No_of_stores_which_sell_that_thing[i] = int(values[train_data.Item_Identifier[i]])
test_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(test_data)):
    test_data.No_of_stores_which_sell_that_thing[i] = int(values[test_data.Item_Identifier[i]])


train_data['Different_Goods_of_this_type'] = np.nan
for i in range(0, len(train_data)):
    train_data.Different_Goods_of_this_type[i] = int(values[train_data.Item_Type[i]])

import random
rand_int = []
for i in range(0, len(X_train_cont_sc)):
    rand_int.append(random.randint(1, 100))
    
min_max_list = MinMaxScaler()
rand_int = min_max_list.fit_transform(np.array(rand_int).reshape(-1, 1))

plt.scatter(x = train_data['No_of_stores_which_sell_that_thing'].tolist(), y = train_data.Item_Outlet_Sales.tolist())
plt.scatter(x = train_data['Different_Goods_of_this_type'].tolist(), y = train_data.Item_Outlet_Sales.tolist())

X_train = train_set.iloc[:, [0,1,2,3,4,6,7,8,10]]
X_valid = valid_set.iloc[:, [0,1,2,3,4,6,7,8,10]]
y_train = np.sqrt(train_set.iloc[:, 9].values)
y_valid = np.sqrt(valid_set.iloc[:, 9].values)

categorical_var = np.where(X_train.dtypes != np.float)[0]
print('\nCategorical Variables indices : ',categorical_var)


from catboost import CatBoostRegressor
regressor = CatBoostRegressor(iterations = 10)
regressor.fit(X_train, y_train, cat_features = [2, 4, 5, 6, 7, 8], plot = False)
y_pred = regressor.predict(X_valid)
y_pred_train = regressor.predict(X_train)

from sklearn.metrics import mean_squared_error
mean_squared_error(np.square(y_valid), np.square(y_pred))
mean_squared_error(np.square(y_train), np.square(y_pred_train))

# Final Predictions

X_train = train_data.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10, 12, 13]]
X_test = test_data.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]]
y_train = np.sqrt(train_data.iloc[:, 11])

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train.iloc[:, [0, 1, 3, 9]] = sc_x.fit_transform(X_train.iloc[:, [0, 1, 3, 9]])
X_test.iloc[:, [0, 1, 3, 9]] = sc_x.transform(X_test.iloc[:, [0, 1, 3, 9]])

regressor = CatBoostRegressor(iterations = 10)
regressor.fit(X_train, y_train, cat_features = [2, 4, 5, 6, 7, 8], plot = False)
y_pred = regressor.predict(X_test)
y_pred_train = regressor.predict(X_train)

Solution = pd.DataFrame(np.square(y_pred))
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_6.csv', index = False)


''' CatBoost gives 1162.91 score '''