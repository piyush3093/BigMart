'''Only XGBoost and basic data preprocessing '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train data.csv')
test_data = pd.read_csv('test data.csv')

for i in range(0, len(train_data)):
    if train_data.Item_Fat_Content[i] == 'low fat':
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'LF'):
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'reg'):
        train_data.Item_Fat_Content[i] = 'Regular'
    else:
        continue

for i in range(0, len(train_data)):
    if (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Low Fat'):
        train_data.Item_Weight[i] = 12.940000
    elif (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Regular'):
        train_data.Item_Weight[i] = 12.710000

train_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')

for i in range(0, len(test_data)):
    if test_data.Item_Fat_Content[i] == 'low fat':
        test_data.Item_Fat_Content[i] = 'Low Fat'
    elif(test_data.Item_Fat_Content[i] == 'LF'):
        test_data.Item_Fat_Content[i] = 'Low Fat'
    elif(test_data.Item_Fat_Content[i] == 'reg'):
        test_data.Item_Fat_Content[i] = 'Regular'
    else:
        continue

test_data.Outlet_Size = test_data.Outlet_Size.fillna('Unknown')

for i in range(0, len(test_data)):
    if (np.isnan(test_data.Item_Weight[i]) and test_data.Item_Fat_Content[i] == 'Low Fat'):
        test_data.Item_Weight[i] = 12.940000
    elif (np.isnan(test_data.Item_Weight[i]) and test_data.Item_Fat_Content[i] == 'Regular'):
        test_data.Item_Weight[i] = 12.710000
        
training_and_testing_X = train_data.drop('Item_Outlet_Sales', axis = 1)
training_and_testing_X = pd.concat([training_and_testing_X, test_data], axis = 0)

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

training_and_testing_X, dummy = encoding(training_and_testing_X, 'Outlet_Type')
training_and_testing_X = training_and_testing_X.drop('Item_Identifier', axis = 1)

X_train = training_and_testing_X.iloc[:8523, :].values
X_test = training_and_testing_X.iloc[8523:, :].values
y_train = train_data.iloc[:, 11].values

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train_3, y_train)

new_data = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis = 1)
corr_mat = new_data.corr()

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X_train_3, y = y_train, cv = 5, scoring = 'neg_mean_squared_error')
score.mean()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_2 = sc.fit_transform(X_train)

from sklearn.decomposition import PCA
pca = PCA().fit(X_train_2)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
explained_variance = pca.explained_variance_ratio_

pca = PCA(n_components = 10)
X_train_3 = pca.fit_transform(X_train_2)

y_train = np.sqrt(y_train)
X_train[:, 2] = np.sqrt(X_train[:, 2])
X_train_3 = pca.fit_transform(X_train_2)

X_test[:, 2] = np.sqrt(X_test[:, 2])
X_test = sc.transform(X_test)
X_test = pca.transform(X_test)

y_pred = regressor.predict(X_test)
new_arr = np.array([1, 2, 3, 4])
new_arr = np.square(new_arr)
y_pred = np.square(y_pred)

Solution = pd.DataFrame(y_pred)
Solution = Solution.rename(columns = {0 : 'Item_Outlet_Sales'})
Solution = pd.concat([test_data['Item_Identifier'], test_data['Outlet_Identifier'], Solution], axis = 1)

Solution.to_csv('Solution.csv', index = False)

'''Score of 1174 '''
''' 175 kcross mean validation '''

training_X, dummy = encoding(training_X, 'Outlet_Type')
training_X = training_X.drop('Item_Identifier', axis = 1)
training_X = training_X.drop('Item_Outlet_Sales', axis = 1)
training_X.Item_Visibility = np.sqrt(training_X.Item_Visibility)
X_train = training_X.iloc[:, :].values
y_train = train_data.loc[:, 'Item_Outlet_Sales'].values
y_train = np.sqrt(y_train)

''' 114.419 kcross score '''
training_and_testing_X = training_and_testing_X.drop('Item_Identifier', axis = 1)

training_and_testing_X['Item_Fat_Content'] = labelencoder.fit_transform(training_and_testing_X['Item_Fat_Content'])
training_and_testing_X['Item_Type'] = labelencoder.fit_transform(training_and_testing_X['Item_Type'])
training_and_testing_X['Outlet_Identifier'] = labelencoder.fit_transform(training_and_testing_X['Outlet_Identifier'])
training_and_testing_X['Outlet_Size'] = labelencoder.fit_transform(training_and_testing_X['Outlet_Size'])
training_and_testing_X['Outlet_Location_Type'] = labelencoder.fit_transform(training_and_testing_X['Outlet_Location_Type'])
training_and_testing_X['Outlet_Type'] = labelencoder.fit_transform(training_and_testing_X['Outlet_Type'])

''' 110 kcross score '''
''' 1490 analytics score '''


