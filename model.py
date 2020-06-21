data_train_X = data_train_X.drop('training_hours', axis = 1)
data_test_X = test_data.drop('training_hours', axis = 1)
data = pd.concat([data_train_X, data_test_X], axis = 0)

def encoding(dataset, feature_to_encode):
    dummies = pd.get_dummies(dataset[feature_to_encode], drop_first = True)
    dataset = dataset.drop([feature_to_encode], axis = 1)
    dataset = pd.concat([dataset, dummies], axis = 1)
    return dataset, dummies

data, dummy = encoding(data, 'company_type')

training_data = data.iloc[:18359, :]
testing_data = data.iloc[18359:, :]
training_data = training_data.drop('city', axis = 1)
testing_data = testing_data.drop('enrollee_id', axis = 1)

training_data.head()
testing_data.head()

X_train = training_data.iloc[:, 2:].values
X_test = testing_data.iloc[:, 2:].values
y_train = train_data.iloc[:, 13].values

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

fs = SelectKBest(score_func = chi2, k = 20)
fs.fit(X_train, y_train)
X_train_new = fs.transform(X_train)
X_test_new = fs.transform(X_test)

X_train_new = np.concatenate([X_train_new, training_data.iloc[:, [0, 1]]], axis = 1)
X_test_new = np.concatenate([X_test_new, testing_data.iloc[:, [0, 1]]], axis = 1)

from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train_new, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train_new, y = y_train, cv = 5)
accuracies.mean()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_new[:, [20, 21]] = sc.fit_transform(X_train_new[:, [20, 21]])
X_test_new[:, [20, 21]] = sc.transform(X_test_new[:, [20, 21]])

y_pred_xgb = classifier.predict(X_test_new)
solution_3 = pd.DataFrame(y_pred_linear)
solution_3[0].value_counts()

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_new, y_train)

y_pred_linear = classifier.predict(X_test_new)
train_data.target.value_counts()

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 11, metric = 'minkowski', p = 2, algorithm = 'auto')
classifier.fit(X_train_new, y_train)


