import pandas as pd
import joblib
from fis import evaluation

# %% load new data

X_test = pd.read_pickle('data_pred/X_test.pkl')
y_test = pd.read_pickle('data_pred/y_test.pkl')
test_race_id = pd.read_pickle('data_pred/test_race_id.pkl')

# %%
# load the model
loaded_model_rf = joblib.load('models/model_random_forest.pkl')
loaded_model_gb = joblib.load('models/model_gradient_boosting.pkl')
loaded_model_lasso = joblib.load('models/model_lasso.pkl')

# %% Test different models

pred_rf = loaded_model_rf.predict_proba(X_test)[:, 1]
pred_gb = loaded_model_gb.predict_proba(X_test)[:, 1]
pred_lasso = loaded_model_lasso.predict_proba(X_test)[:, 1]

predictions = pd.concat(
    [test_race_id.reset_index(drop=True),
     y_test.reset_index(drop=True),
     pd.Series(pred_rf, name="Predicted_rf").reset_index(drop=True),
     pd.Series(pred_gb, name="Predicted_gb").reset_index(drop=True),
     pd.Series(pred_lasso, name="Predicted_lasso").reset_index(drop=True),
     X_test.reset_index(drop=True)], axis=1)

# Sort by race_id and predicted probabilities in descending order by rf
predictions = predictions.sort_values(by=['race_id', 'Predicted_rf'], ascending=[True, False])

# Create a ranking column within each race_id group
predictions['predicted_rank_rf'] = predictions.groupby('race_id')['Predicted_rf'].rank(method="first", ascending=False)

# Mark 1st, 2nd, and 3rd place explicitly
predictions['predicted_podium_rf'] = predictions['predicted_rank_rf'].apply(lambda x: 1 if x <= 3 else 0)

# Sort by race_id and predicted probabilities in descending order by rf
predictions = predictions.sort_values(by=['race_id', 'Predicted_gb'], ascending=[True, False])

# Create a ranking column within each race_id group
predictions['predicted_rank_gb'] = predictions.groupby('race_id')['Predicted_gb'].rank(method="first", ascending=False)

# Mark 1st, 2nd, and 3rd place explicitly
predictions['predicted_podium_gb'] = predictions['predicted_rank_gb'].apply(lambda x: 1 if x <= 3 else 0)

# Sort by race_id and predicted probabilities in descending order by rf
predictions = predictions.sort_values(by=['race_id', 'Predicted_lasso'], ascending=[True, False])

# Create a ranking column within each race_id group
predictions['predicted_rank_lasso'] = predictions.groupby('race_id')['Predicted_lasso'].rank(method="first", ascending=False)

# Mark 1st, 2nd, and 3rd place explicitly
predictions['predicted_podium_lasso'] = predictions['predicted_rank_lasso'].apply(lambda x: 1 if x <= 3 else 0)

# Evaluate the predictions made
eval_rf = evaluation.evaluation(predictions['podium'], predictions['predicted_podium_rf'])
eval_gb = evaluation.evaluation(predictions['podium'], predictions['predicted_podium_gb'])
eval_lasso = evaluation.evaluation(predictions['podium'], predictions['predicted_podium_lasso'])
