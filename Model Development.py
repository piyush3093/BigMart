''' Model Development '''

X_train = train_data.iloc[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]].values
y_train = train_data.iloc[:, 11].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
X_train[:, 1] = labelencoder.fit_transform(X_train[:, 1])
X_train[:, 3] = labelencoder.fit_transform(X_train[:, 3])
X_train[:, 5] = labelencoder.fit_transform(X_train[:, 5])
X_train[:, 7] = labelencoder.fit_transform(X_train[:, 7])
X_train[:, 8] = labelencoder.fit_transform(X_train[:, 8])
X_train[:, 9] = labelencoder.fit_transform(X_train[:, 9])

X_train = X_train.astype('float64')

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train_2, y_train)

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train_2, y_train)

from sklearn.model_selection import cross_val_score
score = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 5, scoring = 'neg_mean_squared_error')
score.mean()
score.std()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_2 = sc.fit_transform(X_train)

from sklearn.decomposition import PCA
pca = PCA().fit(X_train_2)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
explained_variance = np.cumsum(pca.explained_variance_ratio_)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([('encoder', OneHotEncoder(), [42])], remainder = 'passthrough')
X_train = columntransformer.fit_transform(X_train)
X_train = X_train.astype('float64')

onehotencoder = OneHotEncoder(categories = 'auto', sparse = False)
X_train = onehotencoder.fit_transform(X_train)

data_train_X = train_data.copy(deep = True)
data_train_X['Item_Fat_Content'] = labelencoder.fit_transform(data_train_X['Item_Fat_Content'])
data_train_X['Item_Type'] = labelencoder.fit_transform(data_train_X['Item_Type'])
data_train_X['Outlet_Size'] = labelencoder.fit_transform(data_train_X['Outlet_Size'])
data_train_X['Outlet_Location_Type'] = labelencoder.fit_transform(data_train_X['Outlet_Location_Type'])
data_train_X['Outlet_Type'] = labelencoder.fit_transform(data_train_X['Outlet_Type'])
data_train_X['Establishment_Period'] = labelencoder.fit_transform(data_train_X['Establishment_Period'])
data_train_X['Outlet_Identifier'] = labelencoder.fit_transform(data_train_X['Outlet_Identifier'])
data_train_X.Item_Identifier = train_data.Item_Identifier

data_train_X = data_train_X.drop('Item_Identifier', axis = 1)
data_train_X_new = data_train_X.copy(deep = True)
data_train_X = train_data.copy(deep = True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columntransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1, 3, 5, 7, 8, 9, 11])], remainder = 'passthrough')
data_train_X = columntransformer.fit_transform(data_train_X)

onehotencoder = OneHotEncoder(handle_unknown = 'ignore')
data_train_X = onehotencoder.fit_transform(data_train_X.loc[:, ['Item_Type']])

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

data_train_X, dummy = encoding(data_train_X, 'Establishment_Period')

data_train_X = data_train_X.drop('Item_Identifier', axis = 1)
data_train_X = data_train_X.drop('Outlet_Establishment_Year', axis = 1)
data_train_X = data_train_X.drop('Item_Outlet_Sales', axis = 1)

X_train = data_train_X.iloc[:, :].values


X_train = data_train_X.loc[:, 'Item_MRP'].values
X_train = X_train.reshape(-1, 1)

pca = PCA(n_components = 28)
X_train_2 = pca.fit_transform(X_train_2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train_2, y_train)



