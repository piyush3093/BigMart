''' Mean Encoding '''

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

for i in range(0, len(train_data)):
    if (pd.isnull(train_data.Outlet_Size[i]) and train_data.Outlet_Type[i] == 'Grocery Store'):
        train_data.Outlet_Size[i] = 'Small'
    if (pd.isnull(train_data.Outlet_Size[i]) and train_data.Outlet_Type[i] == 'Supermarket Type1'):
        train_data.Outlet_Size[i] = 'Small'
    if (pd.isnull(train_data.Outlet_Size[i]) and train_data.Outlet_Type[i] == 'Supermarket Type2'):
        train_data.Outlet_Size[i] = 'Medium'
    if (pd.isnull(train_data.Outlet_Size[i]) and train_data.Outlet_Type[i] == 'Supermarket Type3'):
        train_data.Outlet_Size[i] = 'Medium'

for i in range(0, len(test_data)):
    if (pd.isnull(test_data.Outlet_Size[i]) and test_data.Outlet_Type[i] == 'Grocery Store'):
        test_data.Outlet_Size[i] = 'Small'
    if (pd.isnull(test_data.Outlet_Size[i]) and test_data.Outlet_Type[i] == 'Supermarket Type1'):
        test_data.Outlet_Size[i] = 'Small'
    if (pd.isnull(test_data.Outlet_Size[i]) and test_data.Outlet_Type[i] == 'Supermarket Type2'):
        test_data.Outlet_Size[i] = 'Medium'
    if (pd.isnull(test_data.Outlet_Size[i]) and test_data.Outlet_Type[i] == 'Supermarket Type3'):
        test_data.Outlet_Size[i] = 'Medium'

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

# Target Encoding the categorical data

def mean_encode(dataset, dataset_2, feature, target):
    group = dataset[target].groupby(dataset[feature]).mean()
    values = group.to_dict()
    for i in range(0, len(dataset)):
        dataset[feature][i] = values[dataset[feature][i]]
    for i in range(0, len(dataset_2)):
        dataset_2[feature][i] = values[dataset_2[feature][i]]
    return dataset, dataset_2

train_data, test_data = mean_encode(train_data, test_data, 'Establishment_Period', 'Item_Outlet_Sales')

train_data.Item_Identifier[train_data.Item_Identifier == 'FDA15'].count()

val_data = pd.Dataframe()

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X_train_sc, y = y_train, cv = 5, scoring = 'neg_mean_squared_error')
score.mean()

train_data.Establishment_Period = train_data.Establishment_Period.astype('object')
train_data.info()
test_data.Establishment_Period = test_data.Establishment_Period.astype('object')

X_train = train_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]].values
X_test = test_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]].values
y_train = train_data.iloc[:, 11].values
X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

X_train[:, 2] = np.sqrt(X_train[:, 2])
X_test[:, 2] = np.sqrt(X_test[:, 2])
y_train = np.sqrt(y_train)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc_target = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

y_train = y_train.reshape(-1, 1)
y_train_sc = sc_target.fit_transform(y_train)

from sklearn.decomposition import PCA
pca = PCA().fit(X_train_sc)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

pca = PCA(n_components = 10)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.fit_transform(X_test_sc)

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train_pca, y_train_sc)
y_pred = regressor.predict(X_test_pca)

y_pred = sc_target.inverse_transform(y_pred)
y_pred = np.square(y_pred)

Solution = pd.DataFrame(y_pred)
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_3.csv', index = False)

''' Score of 1584 '''

y_pred_train = regressor.predict(X_train_pca)

plt.scatter(y_train_sc, y_pred_train)
plt.show()