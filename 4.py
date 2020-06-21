''' Feature Generation and different models '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train data.csv')
test_data = pd.read_csv('test data.csv')

sns.kdeplot(data = train_data.Item_MRP)
sns.distplot(a = test_data.Item_Visibility)
test_data.Item_Visibility.value_counts().head()

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

# Creating New Features
group = train_data.groupby([train_data.Outlet_Type, train_data.Outlet_Identifier], as_index = False).first()

data = pd.concat([train_data.iloc[:, :11], test_data], axis = 0)
values = data.Item_Identifier.value_counts().to_dict()

train_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(train_data)):
    train_data.No_of_stores_which_sell_that_thing[i] = int(values[train_data.Item_Identifier[i]])

test_data['No_of_stores_which_sell_that_thing'] = np.nan
for i in range(0, len(test_data)):
    test_data.No_of_stores_which_sell_that_thing[i] = int(values[test_data.Item_Identifier[i]])

group = data.groupby([data.Item_Identifier, data.Item_Type], as_index = False).first()
group.Item_Identifier.nunique()
values = group.Item_Type.value_counts().to_dict()

train_data['Different_Goods_of_this_type'] = np.nan
for i in range(0, len(train_data)):
    train_data.Different_Goods_of_this_type[i] = int(values[train_data.Item_Type[i]])

test_data['Different_Goods_of_this_type'] = np.nan
for i in range(0, len(test_data)):
    test_data.Different_Goods_of_this_type[i] = int(values[test_data.Item_Type[i]])
    
corr = train_data.corr()

# Encoding of variables

training_set = train_data.drop(['Item_Outlet_Sales', 'Outlet_Establishment_Year'], axis = 1)
testing_set = test_data.drop(['Outlet_Establishment_Year'], axis = 1)

training_set.Item_Visibility = np.sqrt(training_set.Item_Visibility)
testing_set.Item_Visibility = np.sqrt(testing_set.Item_Visibility)

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

training_and_testing_set = pd.concat([training_set, testing_set], axis = 0)
training_and_testing_set, dummy = encoding(training_and_testing_set, 'Outlet_Type')

training_and_testing_set = training_and_testing_set.drop('Item_Identifier', axis = 1)
y_train = train_data.iloc[:, 11]
y_train = np.sqrt(y_train)

from sklearn.feature_selection import SelectKBest, f_regression
selectkbest = SelectKBest(score_func = f_regression, k = 26)
selectkbest.fit(training_and_testing_set.iloc[:8523, 5:], y_train)

training_and_testing_data = selectkbest.transform(training_and_testing_set.iloc[:, 5:])
training_and_testing_cont = training_and_testing_set.iloc[:, :5]

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()

X_train_cont = sc_x.fit_transform(training_and_testing_cont.iloc[:8523, :])
X_test_cont = sc_x.transform(training_and_testing_cont.iloc[8523:, :])
y_train = sc_y.fit_transform(y_train[:].values.reshape(-1, 1))

from sklearn.decomposition import PCA
pca = PCA().fit(X_train_cont)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

explained_variance = np.cumsum(pca.explained_variance_ratio_)

X_train = np.concatenate([X_train_cont, training_and_testing_data[:8523, :]], axis = 1)
X_test = np.concatenate([X_test_cont, training_and_testing_data[8523:, :]], axis = 1)

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

y_pred = sc_y.inverse_transform(y_pred)
y_pred = np.square(y_pred)

Solution = pd.DataFrame(y_pred)
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_4.csv', index = False)





