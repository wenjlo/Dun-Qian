import pickle
import pandas as pd
import numpy as np
from config import numeric_cols,category_cols,target_col

def numeric_transform(x):
    return x**(1/3)

def main():
    test_data = pd.read_csv("./data/test.csv")

    for col in numeric_cols:
        test_data[col] = test_data[col].astype('float32')
        test_data[col] = test_data[col].apply(numeric_transform)
    for col in category_cols:
        test_data[col] = test_data[col].astype('int').astype('category')

    model = pickle.load(open('./model/xgb_catboost_stacking_model', 'rb'))
    prediction = model.predict_proba(test_data[numeric_cols + category_cols])
    test_data[target_col] = prediction[:,1]
    submission_data = test_data[['ID',target_col]]
    submission_data.to_csv('./submission.csv',index=False)
if __name__ == "__main__":
    main()