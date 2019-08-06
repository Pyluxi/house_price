from feature_select import X_train_sfm,X_test_sfm,y_train
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
params = {
    'n_estimators': range(100,1000,200),
    'max_depth': range(10,100,20),
}

rfr = RandomForestRegressor()
grid_search = GridSearchCV(rfr,params,n_jobs=-1,cv=5)
grid_search.fit(X_train_sfm,y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)