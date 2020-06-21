''' Exploratory Data Analysis '''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv('train data.csv')
test_data = pd.read_csv('test data.csv')

train_data.isnull().sum()
test_data.isnull().sum()
train_data.info()
corr_mat = train_data.corr()

len(train_data.Item_Identifier.unique())
train_data.Outlet_Establishment_Year.max()

sns.distplot(a = train_data.Outlet_Establishment_Year)

''' correlation not a big problem '''

train_data.head()

sns.kdeplot(data = np.sqrt(train_data.Item_Outlet_Sales))
train_data.Item_Outlet_Sales.min()
train_data.Item_Outlet_Sales.max()

''' SQRT of target must be predicted '''

train_data.Item_Fat_Content.unique()
for i in range(0, len(train_data)):
    if train_data.Item_Fat_Content[i] == 'low fat':
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'LF'):
        train_data.Item_Fat_Content[i] = 'Low Fat'
    elif(train_data.Item_Fat_Content[i] == 'reg'):
        train_data.Item_Fat_Content[i] = 'Regular'
    else:
        continue
train_data.Outlet_Type.value_counts()

len(train_data.Outlet_Identifier.unique())
corr = train_data.Outlet_Identifier.groupby([train_data.Item_Type, train_data.Outlet_Identifier]).count()
corr.first()
train_data.Item_Type.value_counts()

sns.kdeplot(data = train_data.Item_Weight)
sns.distplot(a = np.sqrt(train_data.Item_Visibility))

train_data.Item_Visibility.value_counts().head()

sum_of_non_zero = 0
count = 0
for i in range(0, len(train_data)):
    if (train_data.Item_Visibility[i] != 0):
        sum_of_non_zero += train_data.Item_Visibility[i]
        count += 1
    else:
        continue

train_data.Item_Visibility = train_data.Item_Visibility.replace(0, 0.070482)
train_data.Item_Outlet_Sales.groupby(train_data.Item_Fat_Content).mean()
group_1.first()

group_1 = train_data.Outlet_Size.groupby(train_data.Outlet_Identifier)
train_data.Outlet_Location_Type.groupby(train_data.Outlet_Identifier).first()

train_data_new = train_data.loc[:, ['Item_Weight', 'Item_Fat_Content']]
train_data_new = train_data_new.dropna()
train_data_new.Item_Weight.groupby(train_data_new.Item_Fat_Content).mean()
train_data_new.Item_Weight.mean()

for i in range(0, len(train_data)):
    if (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Low Fat'):
        train_data.Item_Weight[i] = 12.940000
    elif (np.isnan(train_data.Item_Weight[i]) and train_data.Item_Fat_Content[i] == 'Regular'):
        train_data.Item_Weight[i] = 12.710000
        
train_data.Item_Weight = train_data.Item_Weight.fillna(12.860000)
train_data.Outlet_Size = train_data.Outlet_Size.fillna('Unknown')

plt.plot(train_data.Item_Outlet_Sales.tolist(), '.')
train_data.max()
sns.kdeplot(data = np.sqrt(train_data.Item_Visibility))
train_data.Item_Visibility.mean()
sns.distplot(a = train_data.Item_MRP)
len(train_data.Item_MRP.unique())

value_count_item = train_data.Item_Identifier.value_counts()
group_1 = train_data.Item_MRP.groupby([train_data.Item_Identifier, train_data.Outlet_Identifier, train_data.Item_Outlet_Sales, train_data.Outlet_Location_Type]).min()

''' Use SQRT of item visbility '''

train_data.Outlet_Establishment_Year.max()
train_data['Establishment_Period'] = pd.cut(x = train_data['Outlet_Establishment_Year'], bins = [1984, 1991, 1997, 2003, 2009], labels = ['very old', 'old', 'medium', 'new'])
train_data.Establishment_Period.value_counts()

train_data.Item_Visibility = np.sqrt(train_data.Item_Visibility)
train_data.Item_Outlet_Sales = np.sqrt(train_data.Item_Outlet_Sales)
train_data.duplicated().value_counts()

plt.scatter(x = train_data.Item_MRP.tolist(), y = train_data.Item_Outlet_Sales.tolist())
plt.xlabel('Item_MRP')
plt.ylabel('Item_Outsales')
plt.show()

labelencoder = LabelEncoder()
train_data_new = labelencoder.fit_transform(train_data.iloc[:, 2])
train_data_new = pd.DataFrame(train_data_new)
train_data_new['column2'] = labelencoder.fit_transform(train_data.iloc[:, 4])
onehotenconder = OneHotEncoder()
train_data_new = onehotencoder.fit_transform(train_data_new)

data_train_X.Item_Visibility = data_train_X.Item_Visibility.replace(0, 0.070482)
train_data.Item_Fat_Content.unique()

train_target = np.sqrt(train_data.Item_Outlet_Sales)
train_Item_Visi = train_data.Item_Visibility
train_target.corr(np.sqrt(train_Item_Visi))
train_target.corr(train_Item_Visi/train_data.Item_Weight)


test_data.Outlet_Identifier.unique()
train_data.Outlet_Identifier.unique()

group = train_data.Item_Outlet_Sales.groupby([train_data.Item_Identifier, train_data.Outlet_Identifier]).sum()
group.first()
train_data_new = train_data.copy(deep = True)

train_data_new['no_of_products'] = train_data_new.Item_Outlet_Sales // train_data_new.Item_MRP

# Anova Test

item_and_outlet = train_data[['Item_Type', 'Item_Outlet_Sales']]

grps = pd.unique(item_and_outlet.Item_Type.values)
d_data = {grp : item_and_outlet['Item_Outlet_Sales'][item_and_outlet.Item_Type == grp] for grp in grps}

d_data['Baking Goods']
from scipy import stats
F, p = stats.f_oneway(d_data['Baking Goods'], d_data['Breads'], d_data['Breakfast'])

# Chi-Square Test

contingency_table=pd.crosstab(train_data["Item_Type"], train_data["Outlet_Identifier"])
print('contingency_table :-\n',contingency_table)

#Observed Values
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)
b=stats.chi2_contingency(contingency_table)

Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)

alpha = 0.05
from scipy.stats import chi2
chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)

#p-value
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)
if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")


