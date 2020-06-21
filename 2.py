''' XGBoost, Random Forest, SVR and LightGBM '''

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

# Label and One Hot Encoding the train data
# Selecting best categorical features

train_data_le = train_data.copy(deep = True)
test_data_le = test_data.copy(deep = True)

train_data_cat = train_data_le.drop(['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales', 'Outlet_Establishment_Year'], axis = 1)

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

train_data_cat, dummy = encoding(train_data_cat, 'Establishment_Period')
train_data_cat = train_data_cat.drop('Item_Identifier', axis = 1)
y_train = train_data.iloc[:, 11]
y_train = np.sqrt(y_train)

data = pd.concat([train_data_le, test_data_le], axis = 0)
data = data.drop('Item_Outlet_Sales', axis = 1)
data, dummy = encoding(data, 'Establishment_Period')
data = data.drop(['Item_Identifier', 'Outlet_Establishment_Year'], axis = 1)
data_new = data.copy(deep = True)
data.Item_Visibility = data.Item_Visibility.replace(0, 0.070482)
data.Item_Visibility = np.sqrt(data.Item_Visibility)

sns.kdeplot(data = data.Item_Visibility)
data.Item_Visibility = np.sqrt(data.Item_Visibility)

from sklearn.feature_selection import SelectKBest, f_regression
selectkbest = SelectKBest(score_func = f_regression, k = 26)
selectkbest.fit(data.iloc[:8523, 3:], y_train)
for i in range(len(selectkbest.scores_)):
    print(' %s: %f' % (train_data_cat.columns[i], selectkbest.scores_[i]))
    
data_cat = selectkbest.transform(data.iloc[:, 3:])

data_cont = data.iloc[:, [0, 1, 2]].values

# Selecting Best continous features

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA().fit(train_data_cont_new)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()

explained_variance = np.cumsum(pca.explained_variance_ratio_)

X_train = np.concatenate([data_cont[:8523, :], data_cat[:8523, :]], axis = 1)
X_test = np.concatenate([data_cont[8523:, :], data_cat[8523:, :]], axis = 1)

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train_sc, y_train)
y_pred_xgb = regressor.predict(X_test_sc)
y_pred = np.square(y_pred)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X_train_sc, y = y_train, cv = 5, scoring = 'neg_mean_squared_error')
score.mean()

Solution = pd.DataFrame(y_pred)
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_2.csv', index = False)

''' score 1157, k=26 '''

''' XGBoost kcross = -110.58'''

# Model Testing

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_sc, y_train)
y_pred_linear = regressor.predict(X_test_sc)

'''Linear = -116.47'''

sc_y = StandardScaler()
y_train_sc = sc_y.fit_transform(y_train[:].values.reshape(-1, 1))

from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train_sc, y_train_sc)
y_pred_svr = regressor.predict(X_test_sc)
y_pred_svr = sc_y.inverse_transform(y_pred_svr)

''' SVR - -116.77 '''

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
regressor.fit(X_train_sc, y_train)
y_pred_random_forest = regressor.predict(X_test_sc)

''' Random Forest n = 100, -123.45 '''

from lightgbm import LGBMRegressor
regressor = LGBMRegressor()
regressor.fit(X_train_sc, y_train)
y_pred_lightgbm = regressor.predict(X_test)

''' LightGBM kcross = -114.05'''

y_pred = []
for i in range(0, len(y_pred_xgb)):
    y_pred.append(0.4*y_pred_xgb[i] + 0.3*y_pred_lightgbm[i] + 0.1*y_pred_linear[i] + 0.1*y_pred_random_forest[i] + 0.1*y_pred_svr[i])
    
Solution = pd.DataFrame(np.square(y_pred))
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)
Solution.to_csv('Solution_5.csv', index = False)

''' Bagging score = 1216.88'''

# LightGBM optimization

from sklearn.preprocessing import MinMaxScaler
min_max = MinMaxScaler()
X_train_scaled = min_max.fit_transform(X_train)
X_test_scaled = min_max.transform(X_test)
min_max_y = MinMaxScaler()
y_train_scaled = min_max_y.fit_transform(y_train.iloc[:].values.reshape(-1, 1))

regressor = LGBMRegressor()
regressor.fit(X_train, y_train)
y_pred = regressor.predict
