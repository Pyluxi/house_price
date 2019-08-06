import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer

train_data = pd.read_csv('../tmp/train_1.csv',index_col='Id')
X_test = pd.read_csv('../tmp/test_data_2.csv',index_col='Id')
#标准化
# train_data = (train_data - train_data.mean())/train_data.std()
# print(train_data.shape)
# # print(train_data.describe().T[['min','max']])
# print(train_data.info())
X_train = train_data.iloc[:,:-1]
# print(X_train.head())
# print(X_test.head())
# print(X_train.shape,X_test.shape)
y_train = train_data.iloc[:,-1].values
dict_env = DictVectorizer(sparse=False)#sparse=False表示不产生稀疏矩阵
X_train = dict_env.fit_transform(X_train.to_dict(orient='record'))#orient=record形成列表加字典的形式
# [{column -> value}, … , {column -> value}]的结构
X_test = dict_env.transform(X_test.to_dict(orient='record'))
# X_test = pd.DataFrame(X_test,columns=dict_env.feature_names_)

# print(X_train)
# print(dict_env.feature_names_)
# print(X_train)
# print(y_train)
#使用Random Forest中feature_importances_属性来选择特征
sfm = SelectFromModel(RandomForestRegressor(n_estimators=100,random_state=38),threshold='median')
sfm.fit(X_train,y_train)
X_train_sfm = sfm.transform(X_train)
# print(X_test)
X_test_sfm = sfm.transform(X_test)
# print(sfm.get_support())
# print('基于随机森林进行特征选择后的数据形态：{}'.format(X_test_sfm.shape))
# support = sfm.get_support()
# print(support.shape)
# X_test_sfm = X_test[support]
# print(X_test_sfm.shape)
# print(X_train_sfm.shape)
# print(X_train_sfm)
# print(X_test_sfm)