import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
import joblib
from fis import models
import os

# %% Load data

df = pd.read_pickle('data_clean/final_dataset.pkl')

# %%  Define variables

y_NAME = "podium"
X_NAME_cat = ['gender', 'discipline']
X_NAME_numeric = ['age', 'wcp_last_10', 'wcp_last_5', 'wcp_last_1',
                  'team_wcp_last_10', 'team_wcp_last_5', 'team_wcp_last_1',]
X_NAME_binary = ['own_trainer', 'home_race', 'startmorning', 'startafternoon',
                 'earlyseason', 'midseason', 'top20', 'top21_30',
                 'aut', 'sui', 'ita', 'usa', 'fra', 'ger', 'nor', 'swe',
                 'can', 'slo',]
X_NAME = X_NAME_numeric + X_NAME_cat + X_NAME_binary

# Sort the data by race_id
df_sorted = df.sort_values(by="date")

# Get unique race IDs
unique_race_ids = df_sorted["race_id"].unique()

# Split by time, so training first seasons, test sample last seasons
split_idx = int(len(unique_race_ids) * 0.9)

# Split the data into training and testing ids
train_race_ids = unique_race_ids[:split_idx]
test_race_ids = unique_race_ids[split_idx:]

# Create train and test sets based on race_id membership
train_data = df_sorted[df_sorted["race_id"].isin(train_race_ids)]
test_data = df_sorted[df_sorted["race_id"].isin(test_race_ids)]

# Separate features and target variable
X_train, y_train = train_data[X_NAME], train_data[y_NAME]
X_test, y_test = test_data[X_NAME], test_data[y_NAME]

X_test_race_id = test_data['race_id']


# Define folder paths
pred_folder = "data_pred"
# Save the cleaned data to the clean folder
X_test_save = os.path.join(pred_folder, 'X_test.pkl')
y_test_save = os.path.join(pred_folder, 'y_test.pkl')
X_test_race_id_save = os.path.join(pred_folder, 'test_race_id.pkl')
y_test.to_pickle(y_test_save)
X_test.to_pickle(X_test_save)
X_test_race_id.to_pickle(X_test_race_id_save)

# %% Train different models

# Random Forest

rf = models.classification_model(
    'RandomForest', X_NAME_numeric, X_NAME_cat, X_NAME_binary, **{'n_jobs': -1, 'random_state': 1})

param_grid = models.hyperparameters_grid('RandomForest', True)

# Define folds for cross validation
cv = GroupKFold(n_splits=3)

# Perform GridSearchCV for hyperparameter tuning
grid_search_rf = GridSearchCV(
    estimator=rf, param_grid=param_grid, cv=cv, n_jobs=-1)

# Perform GridSearchCV for hyperparameter tuning by race_id as a group
grid_search_rf.fit(X_train, y_train, groups=train_data["race_id"])

# Save the model
joblib.dump(grid_search_rf, 'models/model_random_forest.pkl')

# Boosting

gb = models.classification_model(
    'GradientBoosting', X_NAME_numeric, X_NAME_cat, X_NAME_binary, **{'random_state': 1})

param_grid = models.hyperparameters_grid('GradientBoosting', True)

# Perform GridSearchCV for hyperparameter tuning
grid_search_gb = GridSearchCV(
    estimator=gb, param_grid=param_grid, cv=cv, n_jobs=-1)
grid_search_gb.fit(X_train, y_train, groups=train_data["race_id"])

# Save the model
joblib.dump(grid_search_gb, 'models/model_gradient_boosting.pkl')

# Lasso

lasso = models.classification_model(
    'Lasso', X_NAME_numeric, X_NAME_cat, X_NAME_binary, **{'random_state': 1})

param_grid = models.hyperparameters_grid('Lasso', True)

# Perform GridSearchCV for hyperparameter tuning
grid_search_lasso = GridSearchCV(
    estimator=lasso, param_grid=param_grid, cv=cv, n_jobs=-1)
grid_search_lasso.fit(X_train, y_train, groups=train_data["race_id"])

# Save the model
joblib.dump(grid_search_lasso, 'models/model_lasso.pkl')

