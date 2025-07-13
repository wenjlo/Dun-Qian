import pickle
import pandas as pd
import xgboost as xgb
import catboost as cat
from best_parameters import xgboost_parameters,catboost_parameters
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from config import numeric_cols,category_cols,target_col

def main():
    data = pd.read_csv("./data/fill_na_data.csv")

    for c in numeric_cols:
        data[c] = data[c].astype('float32')
    for c in category_cols:
        data[c] = data[c].astype('int').astype('category')

    data[target_col] = data[target_col].astype('int')

    x_train = data[numeric_cols+category_cols]
    y_train = data[target_col]

    xgboost = xgb.XGBClassifier(**xgboost_parameters, enable_categorical=True)
    catboost = cat.CatBoostClassifier(**catboost_parameters, cat_features=category_cols)

    xgboost.fit(x_train, y_train)
    catboost.fit(x_train, y_train)

    ensemble_stacking = StackingClassifier(estimators=[('catboost', catboost),
                                                       ('XGBoost', xgboost)],
                                           final_estimator=LogisticRegression())
    ensemble_stacking.fit(x_train, y_train)
    pickle.dump(ensemble_stacking, open('./model/xgb_catboost_stacking_model', 'wb'))
    print('done')

if __name__ == "__main__":
    main()