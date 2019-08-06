from feature_select import X_train_sfm,X_test_sfm,y_train
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd

X_test = pd.read_csv('../data/test.csv')
X_train_sfm = (X_train_sfm - X_train_sfm.mean(axis=0))/X_train_sfm.std(axis=0)
# X_train,X_test,y_train_2,y_test = train_test_split(X_train_sfm,y_train,train_size=0.8)
# print(X_train_sfm)
X_test_sfm = (X_test_sfm - X_test_sfm.mean(axis=0))/X_test_sfm.std(axis=0)

xgb = XGBRegressor()
# xgb.fit(X_train,y_train_2)
# print("模型得分为: {:.2f}".format(xgb.score(X_test,y_test)))
xgb.fit(X_train_sfm,y_train)
y_test = xgb.predict(X_test_sfm)

xgb_submission_1 = pd.DataFrame({'Id':X_test['Id'],'SalePrice':y_test})
print(xgb_submission_1.shape)
xgb_submission_1.to_csv('../tmp/xgb_submission_1.csv',index=None)