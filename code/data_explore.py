import pandas as pd
from function import write_info,missing_count
train_data_file = '../data/train.csv'
test_data_file = '../data/test.csv'
train_data = pd.read_csv(train_data_file,index_col='Id')
test_data = pd.read_csv(test_data_file,index_col='Id')
# print(train_data.isnull())
# print(train_data.head())
#查看是否有缺失值
# print(train_data.shape)
# print(train_data.info())
if __name__ == '__main__':
    # write_info(train_data,'train_data')
    # write_info(test_data,'test_data')
    # na_count = train_data.isnull().sum().sort_values(ascending=False)
    # na_ratio = na_count / len(train_data)
    # na_data = pd.concat([na_count,na_ratio],axis=1,keys=['count','ratio'])
    na_train_data = missing_count(train_data)
    na_test_data = missing_count(test_data)
    # print(na_train_data.head(20))
    #只剩下Electrical特征
    train_data = train_data.drop(na_train_data[na_train_data['count']>1].index,axis=1)
    # print(na_train_data[na_train_data['count']>1].index)
    #查看Electrical最频繁的类别
    # print(train_data['Electrical'].value_counts())
    # na_train_data.to_csv('../tmp/na_train_data.csv')
    # na_test_data.to_csv('../tmp/na_test_data.csv')
    train_data['Electrical'].fillna('SBrkr',inplace=True)
    train_data.to_csv('../tmp/train_1.csv')

#结果显示,LotFrontage、Alley(91)、MasVnrType、MaxVnrArea、BsmtQual、BsmtCond、BsmtExposure、BsmtFinTypel
#BsmtFinType2、Electrical(1)、FireplaceQu(770)、GarageType、GarageYrBlt、GarageFinish、GarageQual
#GarageCond、PoolQC(7)、Fence(281)、MiscFeature(54)
# print(test_data.info())