import pandas as pd
from data_explore import train_data,test_data
from function import missing_count
#其中缺失的比较多的就做丢弃处理
# missing_train_data = train_data[['LotFrontage','MasVnrType','MasVnrArea','BsmtQual',
#                                  'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
#                                  'GarageType','GarageYrBlt','GarageFinish','GarageQual',
#                                  'GarageCond']]
# print(missing_train_data.describe().T[['min','max']])
# train_data = train_data.drop()
na_test_data = missing_count(test_data)
# print(na_test_data.head(35))
#对照train.csv的处理方法，将前18个丢弃
test_data = test_data.drop(na_test_data[na_test_data['count'] > 4].index,axis=1)
# print(missing_count(test_data).head(15))
# na_test_data_2 = missing_count(test_data).head(15)
# na_test_data_2.to_csv('../tmp/na_test_data_2.csv')
# print(test_data['MSZoning'].value_counts())#RL类最多
test_data['MSZoning'].fillna('RL',inplace=True)
# print(test_data['BsmtHalfBath'].value_counts())
test_data['BsmtHalfBath'].fillna(0,inplace=True)
# print(test_data['BsmtFullBath'].value_counts())
test_data['BsmtFullBath'].fillna(0,inplace=True)
# print(test_data['Functional'].value_counts())
test_data['Functional'].fillna('Typ',inplace=True)
# print(test_data['Utilities'].value_counts())
test_data['Utilities'].fillna('AllPub',inplace=True)
# print(test_data['Exterior1st'].value_counts())
test_data['Exterior1st'].fillna('VinylSd',inplace=True)
# print(test_data['KitchenQual'].value_counts())
test_data['KitchenQual'].fillna('TA',inplace=True)
# print(test_data['GarageCars'].value_counts())
test_data['GarageCars'].fillna(2,inplace=True)
# print(test_data['GarageArea'].value_counts())
test_data['GarageArea'].fillna(test_data['GarageArea'].mean(),inplace=True)
# print(test_data['BsmtFinSF1'].value_counts())
test_data['BsmtFinSF1'].fillna(test_data['BsmtFinSF1'].mean(),inplace=True)
# print(test_data['SaleType'].value_counts())
test_data['SaleType'].fillna('WD',inplace=True)
# print(test_data['TotalBsmtSF'].value_counts())
test_data['TotalBsmtSF'].fillna(test_data['TotalBsmtSF'].mean(),inplace=True)
# print(test_data['BsmtUnfSF'].value_counts())
test_data['BsmtUnfSF'].fillna(test_data['BsmtUnfSF'].mean(),inplace=True)
# print(test_data['BsmtFinSF2'].value_counts())
test_data['BsmtFinSF2'].fillna(test_data['BsmtFinSF2'].mean(),inplace=True)
# print(test_data['Exterior2nd'].value_counts())
test_data['Exterior2nd'].fillna('VinylSd',inplace=True)

# print(missing_count(test_data))
test_data.to_csv('../tmp/test_data_2.csv')