import pandas as pd
from config import numeric_cols,category_cols,target_col
from utils.xgb import XGBoostOptimizer
import seaborn as sns
from scipy import stats
import xgboost as xgb
from sklearn.model_selection import train_test_split

data_preproc = pd.read_csv("./data/fill_na_data.csv")

for c in numeric_cols:
    data_preproc[c] = data_preproc[c].astype('float32')
for c in category_cols:
    data_preproc[c] = data_preproc[c].astype('category')

data_preproc[target_col] = data_preproc[target_col].astype('category')

X_train, X_valid, y_train, y_valid = train_test_split(
data_preproc[numeric_cols+category_cols], data_preproc[target_col], test_size=0.2, random_state=42)

reg_optimizer = XGBoostOptimizer(X=X_train, y=y_train, model_type='classifier')
reg_optimizer.optimize_params(
        optuna_n_trials=100,
        num_boost_round_per_trial=1000,
        early_stopping_rounds_per_trial=1
    )
reg_optimizer.train_final_model()