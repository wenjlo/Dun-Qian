import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.samplers import TPESampler



class XGBoostOptimizer:
    def __init__(self,
                 X: pd.DataFrame,
                 y: pd.Series,
                 model_type: str = 'regressor', # 'regressor' or 'classifier'
                 n_splits: int = 5,
                 shuffle_cv: bool = True,
                 random_state_cv: int = 42):
        """
        初始化 XGBoostOptimizer 類別。

        Args:
            X (pd.DataFrame): 特徵資料。
            y (pd.Series): 目標變數。
            model_type (str): 模型類型，'regressor' (迴歸) 或 'classifier' (分類)。
            n_splits (int): 交叉驗證的折數 (K)。
            shuffle_cv (bool): 是否在交叉驗證前打亂資料。
            random_state_cv (int): 交叉驗證的隨機種子，用於重現性。
        """
        self.X = X
        self.y = y
        self.model_type = model_type
        self.n_splits = n_splits
        self.shuffle_cv = shuffle_cv
        self.random_state_cv = random_state_cv

        self.best_params = None
        self.best_score = None
        self.study = None
        self.final_model = None
        self.n_classes = None # For classifier
        self.label_encoder = None # For classifier if y is not numeric

        self._preprocess_target()

    def _preprocess_target(self):

        if self.model_type == 'classifier' and not pd.api.types.is_numeric_dtype(self.y):
            self.label_encoder = LabelEncoder()
            self.y = pd.Series(self.label_encoder.fit_transform(self.y),
                                index=self.y.index, name=self.y.name)
            self.n_classes = len(self.label_encoder.classes_)
            print(f"Encoded target labels for classification: {self.label_encoder.classes_} -> {np.arange(self.n_classes)}")
        elif self.model_type == 'classifier':
            self.n_classes = self.y.nunique()

    def _objective(self, trial):


        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'n_jobs': -1
        }

        # 根據模型類型設置 objective 和 eval_metric
        if self.model_type == 'regressor':
            param['objective'] = 'reg:squarederror'
            param['eval_metric'] = 'rmse'
            metric_to_optimize = 'rmse'
        elif self.model_type == 'classifier':
            if self.n_classes > 2:
                param['objective'] = 'multi:softprob'
                param['num_class'] = self.n_classes
                param['eval_metric'] = 'mlogloss'
                metric_to_optimize = 'mlogloss'
            else:
                param['objective'] = 'binary:logistic'
                param['eval_metric'] = 'logloss'
                metric_to_optimize = 'logloss'
        else:
            raise ValueError("model_type must be 'regressor' or 'classifier'.")


        if self.model_type == 'classifier':
            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=self.shuffle_cv, random_state=self.random_state_cv)
        else:
            kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle_cv, random_state=self.random_state_cv)

        # 儲存每一折的驗證指標
        cv_scores = []
        best_iterations = [] # 儲存每折的最佳迭代次數

        for fold, (train_index, val_index) in enumerate(kf.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]

            dtrain = xgb.DMatrix(X_train, label=y_train,enable_categorical=True)
            dval = xgb.DMatrix(X_val, label=y_val,enable_categorical=True)

            watchlist = [(dtrain, 'train'), (dval, 'eval')]

            # 訓練模型
            model = xgb.train(
                param,
                dtrain,
                num_boost_round=trial.user_attrs.get('num_boost_round', 500), # 使用外部設定的 max rounds
                evals=watchlist,
                early_stopping_rounds=trial.user_attrs.get('early_stopping_rounds', 50), # 使用外部設定的 early stopping rounds
                verbose_eval=False # 不在 Optuna 試驗中打印詳細進度
            )
            score = None
            # 獲取評估分數
            if self.model_type == 'regressor':
                preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1))
                score = np.sqrt(mean_squared_error(y_val, preds)) # RMSE
            elif self.model_type == 'classifier':
                preds = model.predict(dval, iteration_range=(0, model.best_iteration + 1)) # 取得機率

                if self.n_classes > 2:
                    # Multi-class
                    preds = np.argmax(preds, axis=1)
                    score = accuracy_score(y_val, preds)

                else:
                    # Binary
                    preds = np.array(preds)
                    preds = (preds >= 0.5).astype(int)
                    score = roc_auc_score(y_val, preds)

            cv_scores.append(score)
            best_iterations.append(model.best_iteration)

        # 儲存平均最佳迭代次數到 trial 的 user_attrs 中，以便後續訓練最終模型
        trial.set_user_attr("mean_best_iteration", np.mean(best_iterations))

        # 返回要最小化的平均分數 (RMSE 或 LogLoss)
        return np.mean(cv_scores)

    def optimize_params(self,
                        optuna_n_trials: int = 50,
                        optuna_timeout: int = None,
                        optuna_sampler_seed: int = 42,
                        num_boost_round_per_trial: int = 500,
                        early_stopping_rounds_per_trial: int = 50,
                        study_name: str = "xgboost_optimization",
                        storage_path: str = None):
        """
        使用 Optuna 執行超參數優化。

        Args:
            optuna_n_trials (int): Optuna 進行的試驗次數。
            optuna_timeout (int): Optuna 優化的最大運行時間（秒）。
            optuna_sampler_seed (int): Optuna 採樣器的隨機種子。
            num_boost_round_per_trial (int): 每個 Optuna 試驗中 XGBoost 的最大提升迭代次數。
            early_stopping_rounds_per_trial (int): 每個 Optuna 試驗中 XGBoost 的提前停止輪數。
            study_name (str): Optuna Study 的名稱。
            storage_path (str): Optuna 資料庫儲存路徑，例如 "sqlite:///db.sqlite3"。
                                設定後可以從上次中斷的地方繼續優化。
        """
        sampler = TPESampler(seed=optuna_sampler_seed)

        # 設置 Optuna Study
        self.study = optuna.create_study(
            direction='minimize', # 我們希望最小化 RMSE 或 LogLoss
            sampler=sampler,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=True
        )

        # 將 max_rounds 和 early_stopping_rounds 儲存到 study 的 user_attrs 中，供 objective 函數讀取
        self.study.set_user_attr("num_boost_round", num_boost_round_per_trial)
        self.study.set_user_attr("early_stopping_rounds", early_stopping_rounds_per_trial)

        print(f"Starting Optuna optimization for {optuna_n_trials} trials...")
        self.study.optimize(self._objective, n_trials=optuna_n_trials, timeout=optuna_timeout)

        self.best_params = self.study.best_trial.params
        self.best_score = self.study.best_value
        self.best_num_boost_round = int(self.study.best_trial.user_attrs.get("mean_best_iteration", num_boost_round_per_trial))

        print("\n--- Optuna Optimization Finished ---")
        print(f"Best trial number: {self.study.best_trial.number}")
        print(f"Best score ({self.study.direction.name}): {self.best_score:.4f}")
        print(f"Mean best boosting rounds: {self.best_num_boost_round}")
        print("Best parameters:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

    def train_final_model(self):
        """
        使用最佳參數在所有訓練資料上訓練最終的 XGBoost 模型。
        """
        if self.best_params is None:
            raise RuntimeError("Please run optimize_params() first to find the best parameters.")

        print("\n--- Training Final XGBoost Model with Best Parameters ---")

        # 調整 eval_metric 和 objective 以匹配 DMatrix
        final_params = self.best_params.copy()
        if self.model_type == 'regressor':
            final_params['objective'] = 'reg:squarederror'
            final_params['eval_metric'] = 'rmse' # 只是為了打印，實際訓練不依賴 eval_metric
        elif self.model_type == 'classifier':
            if self.n_classes > 2:
                final_params['objective'] = 'multi:softprob'
                final_params['num_class'] = self.n_classes
                final_params['eval_metric'] = 'mlogloss'
            else:
                final_params['objective'] = 'binary:logistic'
                final_params['eval_metric'] = 'logloss'

        dtrain_final = xgb.DMatrix(self.X, label=self.y,enable_categorical=True)

        self.final_model = xgb.train(
            final_params,
            dtrain_final,
            num_boost_round=self.best_num_boost_round,
            verbose_eval=100 # 可以打印進度
        )
        print("最終模型訓練完成！")

    def predict(self, X_new: pd.DataFrame):
        """
        使用訓練好的最終模型進行預測。
        """
        if self.final_model is None:
            raise RuntimeError("Please train the final model using train_final_model() first.")

        dtest = xgb.DMatrix(X_new,enable_categorical=True)
        predictions = self.final_model.predict(dtest)

        if self.model_type == 'classifier':
            if self.n_classes >2:
                predictions = np.argmax(predictions,axis=1)
            else:
                predictions = np.array(predictions)
                predictions = (predictions >= 0.5).astype(int)

        return predictions