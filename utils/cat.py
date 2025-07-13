import optuna
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import logging
import sys


optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))


class CatBoostOptimizer:
    """
    A class to optimize CatBoost hyperparameters using Optuna and K-Fold Cross-Validation.

    Attributes:
        X (pd.DataFrame or np.ndarray): Features for training.
        y (pd.Series or np.ndarray): Target variable.
        model_type (str): Type of model, 'classifier' or 'regressor'.
        categorical_features (list): List of categorical feature names or indices.
        best_params (dict): Dictionary of the best hyperparameters found by Optuna.
        best_model (CatBoostClassifier/CatBoostRegressor): The final CatBoost model
                                                          trained with best_params on the full dataset.
    """

    def __init__(self, X, y, model_type, categorical_features=None):
        """
        Initializes the CatBoostOptimizer.

        Args:
            X (pd.DataFrame or np.ndarray): Features for training.
            y (pd.Series or np.ndarray): Target variable.
            model_type (str): Must be 'classifier' or 'regressor'.
            categorical_features (list, optional): List of categorical feature names (if X is DataFrame)
                                                   or indices (if X is NumPy array). Defaults to None.
        Raises:
            ValueError: If model_type is not 'classifier' or 'regressor'.
        """
        if model_type not in ['classifier', 'regressor']:
            raise ValueError("model_type must be 'classifier' or 'regressor'.")

        self.X = X
        self.y = y
        self.model_type = model_type
        self.categorical_features = categorical_features
        self.best_params = None
        self.best_model = None
        self.study = None  # Store the Optuna study object

        # Determine CatBoost model class
        self.catboost_model_class = CatBoostClassifier if self.model_type == 'classifier' else CatBoostRegressor

    def _objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization.
        This function performs K-Fold cross-validation for a given set of hyperparameters.

        Args:
            trial (optuna.Trial): A trial object from Optuna.

        Returns:
            float: The average evaluation metric across all cross-validation folds.
        """
        # Define the hyperparameter search space
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_seed': 42,
            'verbose': 0,  # Suppress verbose output during trials
            'early_stopping_rounds': 50  # Early stopping for each fold
        }

        # Conditional hyperparameters based on grow_policy
        grow_policy = trial.suggest_categorical('grow_policy', ['SymmetricTree', 'Depthwise', 'Lossguide'])
        params['grow_policy'] = grow_policy

        if grow_policy == 'Lossguide':
            params['min_data_in_leaf'] = trial.suggest_int('min_data_in_leaf', 1, 100)
            params['max_leaves'] = trial.suggest_int('max_leaves', 2, 64)  # Max leaves for Lossguide

        # Choose cross-validation splitter based on model type
        if self.model_type == 'classifier':
            cv_splitter = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            # Define evaluation metric for Optuna
            if self.metric == 'auto':
                params['eval_metric'] = 'Logloss'
                score_func = log_loss
                direction_multiplier = 1  # Minimize logloss
            elif self.metric == 'accuracy':
                params['eval_metric'] = 'Accuracy'
                score_func = accuracy_score
                direction_multiplier = -1  # Maximize accuracy
            elif self.metric == 'roc_auc':
                params['eval_metric'] = 'AUC'
                score_func = roc_auc_score
                direction_multiplier = -1  # Maximize AUC
            else:
                params['eval_metric'] = self.metric  # Use user-defined metric
                score_func = self._get_sklearn_metric_func(self.metric)
                direction_multiplier = 1 if self.direction == 'minimize' else -1

            # For multi-class classification, use 'MultiClass' or 'MultiClassOneVsAll'
            if len(np.unique(self.y)) > 2 and self.model_type == 'classifier':
                params['loss_function'] = 'MultiClass'  # Or 'MultiClassOneVsAll'
            else:
                params['loss_function'] = 'Logloss'  # Binary classification
        else:  # Regressor
            cv_splitter = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            # Define evaluation metric for Optuna
            if self.metric == 'auto':
                params['eval_metric'] = 'RMSE'
                score_func = mean_squared_error
                direction_multiplier = 1  # Minimize RMSE
            elif self.metric == 'mae':
                params['eval_metric'] = 'MAE'
                score_func = mean_absolute_error
                direction_multiplier = 1  # Minimize MAE
            else:
                params['eval_metric'] = self.metric  # Use user-defined metric
                score_func = self._get_sklearn_metric_func(self.metric)
                direction_multiplier = 1 if self.direction == 'minimize' else -1

            params['loss_function'] = 'RMSE'  # Default for regression

        fold_scores = []

        # Perform K-Fold Cross-Validation
        for fold, (train_idx, val_idx) in enumerate(cv_splitter.split(self.X, self.y)):
            X_train, X_val = self.X.iloc[train_idx] if isinstance(self.X, pd.DataFrame) else self.X[train_idx], \
                self.X.iloc[val_idx] if isinstance(self.X, pd.DataFrame) else self.X[val_idx]
            y_train, y_val = self.y.iloc[train_idx] if isinstance(self.y, pd.Series) else self.y[train_idx], \
                self.y.iloc[val_idx] if isinstance(self.y, pd.Series) else self.y[val_idx]

            # Create CatBoost Pool objects for efficient data handling, especially with categorical features
            train_pool = Pool(X_train, y_train, cat_features=self.categorical_features)
            val_pool = Pool(X_val, y_val, cat_features=self.categorical_features)

            model = self.catboost_model_class(**params)

            try:
                model.fit(train_pool,
                          eval_set=val_pool,
                          verbose=False)  # Suppress verbose output for each fold

                # Make predictions for evaluation
                if self.model_type == 'classifier':
                    if self.metric == 'accuracy':
                        preds = model.predict(X_val)
                    else:  # For Logloss, AUC, etc., use predict_proba
                        preds = model.predict_proba(X_val)
                        if len(np.unique(self.y)) > 2:  # Multi-class
                            # Ensure preds have correct shape for log_loss (n_samples, n_classes)
                            # CatBoost's predict_proba for multi-class returns (n_samples, n_classes)
                            pass
                        elif preds.ndim > 1 and preds.shape[1] == 2:  # Binary, take probability of positive class
                            preds = preds[:, 1]
                        elif preds.ndim == 1:  # Binary, already 1D array of probabilities
                            pass
                        else:
                            raise ValueError(f"Unexpected prediction shape for classifier: {preds.shape}")
                else:  # Regressor
                    preds = model.predict(X_val)

                # Calculate score
                if self.model_type == 'classifier' and self.metric == 'accuracy':
                    score = score_func(y_val, preds)
                elif self.model_type == 'classifier' and self.metric == 'roc_auc':
                    # roc_auc_score requires probabilities for binary, or one-vs-rest for multi-class
                    if len(np.unique(self.y)) > 2:
                        score = roc_auc_score(y_val, preds, multi_class='ovr')
                    else:
                        score = roc_auc_score(y_val, preds)
                elif self.model_type == 'classifier' and self.metric == 'logloss':
                    score = log_loss(y_val, preds)
                elif self.model_type == 'regressor' and self.metric == 'rmse':
                    score = np.sqrt(score_func(y_val, preds))
                elif self.model_type == 'regressor' and self.metric == 'mae':
                    score = score_func(y_val, preds)
                else:  # Fallback for custom metric if not explicitly handled
                    # This requires the user to ensure the score_func handles preds correctly
                    score = score_func(y_val, preds)

                fold_scores.append(score)

            except Exception as e:
                print(f"Error during CatBoost fit/predict in trial {trial.number}, fold {fold}: {e}")
                # If an error occurs, report a very high (bad) score to Optuna
                # This helps Optuna avoid problematic parameter combinations
                return float('inf') if self.direction == 'minimize' else float('-inf')

            # Optuna Pruning: Report intermediate value to the trial
            trial.report(np.mean(fold_scores) * direction_multiplier, fold)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # Return the average score across all folds, multiplied by direction_multiplier
        # Optuna by default minimizes, so if we're maximizing (e.g., accuracy), we return negative score
        return np.mean(fold_scores) * direction_multiplier

    def _get_sklearn_metric_func(self, metric_name):
        """Helper to get sklearn metric function based on name."""
        if metric_name == 'accuracy':
            return accuracy_score
        elif metric_name == 'roc_auc':
            return roc_auc_score
        elif metric_name == 'logloss':
            return log_loss
        elif metric_name == 'rmse':
            return mean_squared_error  # RMSE will be sqrt'd in objective
        elif metric_name == 'mae':
            return mean_absolute_error
        else:
            raise ValueError(
                f"Unsupported metric: {metric_name}. Please choose from 'accuracy', 'roc_auc', 'logloss', 'rmse', 'mae' or provide a custom objective.")

    def optimize(self, n_trials, n_splits=5, random_state=42, metric='auto', direction='auto'):
        """
        Runs the Optuna hyperparameter optimization.

        Args:
            n_trials (int): Number of trials (hyperparameter combinations) for Optuna to explore.
            n_splits (int, optional): Number of folds for cross-validation. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to 42.
            metric (str, optional): Evaluation metric to optimize.
                                    'auto' (default): 'Logloss' for classifier, 'RMSE' for regressor.
                                    For classifier: 'accuracy', 'roc_auc', 'logloss'.
                                    For regressor: 'rmse', 'mae'.
            direction (str, optional): Optimization direction.
                                       'auto' (default): 'minimize' for 'Logloss', 'RMSE', 'MAE';
                                       'maximize' for 'accuracy', 'roc_auc'.
                                       Can be explicitly 'minimize' or 'maximize'.
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.metric = metric

        # Auto-detect direction if not explicitly provided
        if direction == 'auto':
            if self.model_type == 'classifier':
                if metric in ['accuracy', 'roc_auc']:
                    self.direction = 'maximize'
                else:  # Default to minimize for Logloss or other loss metrics
                    self.direction = 'minimize'
            else:  # Regressor
                self.direction = 'minimize'  # RMSE, MAE are minimized
        else:
            self.direction = direction

        print(f"Starting Optuna optimization for CatBoost ({self.model_type})...")
        print(f"Optimizing for metric: '{self.metric}' with direction: '{self.direction}'")
        print(f"Using {self.n_splits}-fold cross-validation.")

        self.study = optuna.create_study(direction=self.direction,
                                         sampler=optuna.samplers.TPESampler(seed=self.random_state),
                                         pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10))

        self.study.optimize(self._objective, n_trials=n_trials, callbacks=[self._optuna_callback])

        self.best_params = self.study.best_trial.params
        print("\nOptimization finished!")
        print(f"Best trial number: {self.study.best_trial.number}")
        print(f"Best score ({self.metric}): {self.study.best_value * (1 if self.direction == 'minimize' else -1):.4f}")
        print("Best hyperparameters found:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")

    def _optuna_callback(self, study, trial):
        """Callback to print trial results."""
        if trial.state == optuna.trial.TrialState.COMPLETE:
            print(
                f"Trial {trial.number} finished with {self.metric} = {trial.value * (1 if self.direction == 'minimize' else -1):.4f} and params: {trial.params}")
        elif trial.state == optuna.trial.TrialState.PRUNED:
            print(f"Trial {trial.number} pruned.")

    def train_best_model(self):
        """
        Trains the final CatBoost model using the best hyperparameters found
        on the entire dataset (X, y).
        """
        if self.best_params is None:
            print("No best parameters found. Please run optimize() first.")
            return

        print("\nTraining final CatBoost model with best hyperparameters on the full dataset...")
        final_model_params = self.best_params.copy()
        final_model_params['verbose'] = 100  # Show progress for the final model
        final_model_params['random_seed'] = self.random_state  # Ensure reproducibility

        # Set loss_function explicitly for the final model if not already in best_params
        if self.model_type == 'classifier':
            if 'loss_function' not in final_model_params:
                if len(np.unique(self.y)) > 2:
                    final_model_params['loss_function'] = 'MultiClass'
                else:
                    final_model_params['loss_function'] = 'Logloss'
            if 'eval_metric' not in final_model_params:
                final_model_params['eval_metric'] = 'Logloss'  # Default eval metric for final model
        else:
            if 'loss_function' not in final_model_params:
                final_model_params['loss_function'] = 'RMSE'
            if 'eval_metric' not in final_model_params:
                final_model_params['eval_metric'] = 'RMSE'  # Default eval metric for final model

        self.best_model = self.catboost_model_class(**final_model_params)

        # Create a Pool for the full dataset
        full_data_pool = Pool(self.X, self.y, cat_features=self.categorical_features)

        self.best_model.fit(full_data_pool)
        print("Final CatBoost model trained successfully!")

    def get_best_params(self):
        """
        Returns the best hyperparameters found by Optuna.

        Returns:
            dict: The best hyperparameters.
        """
        return self.best_params

    def get_best_model(self):
        """
        Returns the final CatBoost model trained with the best hyperparameters.

        Returns:
            CatBoostClassifier or CatBoostRegressor: The trained best model.
        """
        return self.best_model

    def plot_optimization_history(self):
        """
        Plots the optimization history of the Optuna study.
        Requires 'matplotlib' and 'plotly' to be installed.
        """
        if self.study is None:
            print("No optimization study found. Please run optimize() first.")
            return
        try:
            import plotly  # Optuna plotting relies on plotly
            fig = optuna.visualization.plot_optimization_history(self.study)
            fig.show()
        except ImportError:
            print("Please install plotly for visualization: pip install plotly")

    def plot_param_importances(self):
        """
        Plots the hyperparameter importances from the Optuna study.
        Requires 'matplotlib' and 'plotly' to be installed.
        """
        if self.study is None:
            print("No optimization study found. Please run optimize() first.")
            return
        try:
            import plotly  # Optuna plotting relies on plotly
            fig = optuna.visualization.plot_param_importances(self.study)
            fig.show()
        except ImportError:
            print("Please install plotly for visualization: pip install plotly")