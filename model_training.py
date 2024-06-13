# %%
import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from joblib import Memory
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    SGDRegressor,
)
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    cross_val_score,
    train_test_split,
    RepeatedKFold,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
import torch
import xgboost as xgb

os.environ["TOKENIZERS_PARALLELISM"] = "false"

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# %%
merged_df = pd.read_pickle(r"merged_df.pickle")

X = merged_df.iloc[:, 5:len(merged_df.columns)-1]
y = merged_df['LA_RAPM']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
memory = Memory(location="cache", verbose=0)

estimators = {
    # "randomforestregressor": RandomForestRegressor(),
    "xgboost": xgb.XGBRegressor(),
    "sgdregressor": SGDRegressor(),
    "lasso": Lasso(),
    "ridge": Ridge(),
}

param_grid = {
    # "randomforestregressor": {
    #     "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
    #     "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     "min_samples_split": [2, 5, 10],
    #     "min_samples_leaf": [1, 2, 4],
    #     "bootstrap": [True, False],
    #     "max_features": np.arange(1,150, 10),
    # },
    "xgboost" : {
        'max_depth': [3, 5, 6, 10, 15, 20],
        'learning_rate': np.arange(0.01, 0.3, 0.1),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.4, 1.0, 0.1),
         'colsample_bylevel': np.arange(0.4, 1.0, 0.1),
         'n_estimators': [100, 500, 1000]
    },
    'sgdregressor' : {
                "penalty": ['l1', 'l2', 'elasticnet'], \
                "alpha": np.logspace(-4, 0, 30),\
                "l1_ratio": list(np.arange(0, 1, 0.1)),
                "learning_rate": ['optimal', 'adaptive', 'invscaling'],
                "eta0": list(np.arange(0.01, 1, 0.1)),
                "tol": list(np.arange(0.0001, 0.01, 0.001)),
                "max_iter": list(np.arange(50, 1000, 100))
                                    },
                                    
    "lasso": {
        "alpha":  np.logspace(-4, 0, 30),
        "selection": ['cyclic', 'random'],
        "tol": list(np.arange(0.0001, 0.01, 0.001)),
        "max_iter": list(np.arange(50, 1000, 100))
    },
    "ridge": {
        "alpha": np.logspace(-4, 0, 30),
        "solver": ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
        "tol": list(np.arange(0.0001, 0.01, 0.001))}
}

# %%
# Initialize dictionaries and lists to store results
scores = {}
mse_values = []
r2_values = []

# Predefined pipeline for Yeo-Johnson transformation
pipe_yeo_johnson = make_pipeline(PowerTransformer(method="yeo-johnson", standardize=True))

# Iterate through each regressor in the estimators dictionary
for regressor_name, regressor in estimators.items():
    print(f"Training {regressor_name}...")
    
    # K-Fold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=40)
    
    for train_index, test_index in kfold.split(X_train, y_train):
        X_train_kfold, X_test_kfold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_kfold, y_test_kfold = y_train.iloc[train_index], y_train.iloc[test_index]

        columns = X_train_kfold.columns
        skews, skew_after_transform, diff_skews = [], [], []
        
        # Transform and check skewness of each feature
        for column in columns:
            skews.append((column, X_train_kfold[column].skew()))
            pt = PowerTransformer(method="yeo-johnson")
            data = pt.fit_transform(X_train_kfold[column].values.reshape(-1, 1)).flatten()
            skew_after_transform.append((column, pd.Series(data).skew()))
            diff_skews.append((column, pd.Series(data).skew() - X_train_kfold[column].skew()))

        # Select columns to transform
        columns_transform = [x[0] for x in diff_skews if abs(x[1]) >= 0.5]
        
        preprocessor = ColumnTransformer(
            transformers=[("yeo-johnson", pipe_yeo_johnson, columns_transform)],
            remainder="passthrough"
        )

        print(f"Columns are transformed for {regressor_name}")
        
        # Apply preprocessing
        X_train_kfold = preprocessor.fit_transform(X_train_kfold)
        X_test_kfold = preprocessor.transform(X_test_kfold)

        # Recursive Feature Elimination (RFE)
        rfe = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=100)
        rfe.fit(X_train_kfold, y_train_kfold)
        selected_features = np.where(rfe.support_)[0]

        X_train_kfold = pd.DataFrame(X_train_kfold).iloc[:, selected_features]
        X_test_kfold = pd.DataFrame(X_test_kfold).iloc[:, selected_features]

        print(f"RFE is done for {regressor_name}")

        # Hyperparameter tuning with RandomizedSearchCV
        rand_search = RandomizedSearchCV(
            regressor,
            param_distributions=param_grid[regressor_name],
            n_iter=60,
            cv=5,
            scoring='r2',
            refit='r2',
            error_score='raise',
            n_jobs=5,
            verbose=0,
            random_state=40,
        )
        rand_search.fit(X_train_kfold, y_train_kfold)
        y_pred = rand_search.predict(X_test_kfold)

        print(f"RandomizedSearchCV is done for {regressor_name}")

        mse = mean_squared_error(y_test_kfold, y_pred)
        r2 = r2_score(y_test_kfold, y_pred)

        mse_values.append(mse)
        r2_values.append(r2)

        print(f"\nThe best estimator across ALL searched params:\n{rand_search.best_estimator_}")
        print(f"\nThe best score across ALL searched params:\n{rand_search.best_score_}")
        print(f"\nThe best parameters across ALL searched params:\n{rand_search.best_params_}")

    # Calculate mean MSE and R² for the regressor
    scores[regressor_name] = [np.mean(mse_values), np.mean(r2_values)]

    print(f"Best {regressor_name} parameters: {rand_search.best_params_}")
    print(f"Best {regressor_name} score: {rand_search.best_score_}")

# Display final scores
for regressor_name, score in scores.items():
    print(f"{regressor_name}: MSE = {score[0]:.4f}, R² = {score[1]:.4f}")



