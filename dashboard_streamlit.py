import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from fis.evaluation import evaluation

# Import data
df = pd.read_pickle('data_clean/final_dataset.pkl')

# Convert 'date' column to datetime
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df['season'] = df.apply(lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1, axis=1)
df_2025_and_beyond = df[df.season >= 2024].copy()

# Streamlit app
st.title("Ski World Cup Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
selected_tab = st.sidebar.radio("Go to:", ["Descriptive Statistics", "Predictions & Actual Results"])

if selected_tab == "Descriptive Statistics":
    st.header("Descriptive Statistics")
    
    # Season selector
    season = st.selectbox("Select Season", sorted(df["season"].unique()), index=len(df["season"].unique())-1)
    
    # Filter data for the selected season
    filtered_df = df[df["season"] == season].copy()
    
    # Male and Female Leaderboards
    male_df = filtered_df[filtered_df["gender"] == "men"].groupby("name", as_index=False)["wcp"].sum()
    female_df = filtered_df[filtered_df["gender"] == "women"].groupby("name", as_index=False)["wcp"].sum()
    
    top_males = male_df.nlargest(10, "wcp")
    top_females = female_df.nlargest(10, "wcp")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Male Athletes")
        st.plotly_chart(px.bar(top_males, x="wcp", y="name", orientation="h", title=f"Top 10 Males {season}"))
    with col2:
        st.subheader("Top 10 Female Athletes")
        st.plotly_chart(px.bar(top_females, x="wcp", y="name", orientation="h", title=f"Top 10 Females {season}"))

    # Athlete performance graph
    athlete = st.selectbox("Select an athlete", df["name"].unique())
    if athlete:
        athlete_df = df[df["name"] == athlete]
        yearly_points = athlete_df.groupby("season")["wcp"].sum().reset_index()
        st.plotly_chart(px.line(yearly_points, x="season", y="wcp", title=f"{athlete}'s World Cup Points"))

elif selected_tab == "Predictions & Actual Results":
    st.header("Predictions & Actual Results")
    
    race_id = st.selectbox("Select Race", df_2025_and_beyond['race_id'].unique())
    model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Lasso"])
    
    # Load model
    model_map = {"Random Forest": "models/model_random_forest.pkl", "Gradient Boosting": "models/model_gradient_boosting.pkl", "Lasso": "models/model_lasso.pkl"}
    model = joblib.load(model_map[model_choice])
    
    # Get race data
    race_data = df_2025_and_beyond[df_2025_and_beyond['race_id'] == race_id].copy()
    # Feature selection
    X_NAME_cat = ['gender', 'discipline']
    X_NAME_numeric = ['age', 'wcp_last_10', 'wcp_last_5', 'wcp_last_1',
                      'team_wcp_last_10', 'team_wcp_last_5', 'team_wcp_last_1']
    X_NAME_binary = ['own_trainer', 'home_race', 'startmorning', 'startafternoon',
                     'earlyseason', 'midseason', 'top20', 'top21_30',
                     'aut', 'sui', 'ita', 'usa', 'fra', 'ger', 'nor', 'swe',
                     'can', 'slo']
    X_NAME = X_NAME_numeric + X_NAME_cat + X_NAME_binary
    race_data['Predicted'] = model.predict_proba(race_data[X_NAME])[:, 1]
    predictions = race_data.sort_values(by='Predicted', ascending=False).head(3)
    
    # Display Predictions
    st.subheader("Predicted Podium")
    st.plotly_chart(px.bar(predictions, x='name', y='Predicted', title="Top 3 Predicted Athletes"))
    
    # Actual Results
    actual_results = df[(df['race_id'] == race_id) & (df['total_rank'] <= 3)]
    st.subheader("Actual Podium")
    st.plotly_chart(px.bar(actual_results, x='name', y='total_rank', title="Actual Top 3 Finishers"))
    
    # Model Evaluation Metrics
    df_test = df[df['date'] >= pd.Timestamp('2024-03-09')]
    df_test['Predicted'] = model.predict_proba(df_test[X_NAME])[:, 1]
    df_test = df_test.sort_values(by='Predicted', ascending=False)
    df_test['predicted_podium'] = df_test['Predicted'].rank(method="first", ascending=False) <= 3
    
    metrics = evaluation(df_test['podium'], df_test['predicted_podium'])
    st.subheader("Model Evaluation Metrics")
    st.write(f"Accuracy: {metrics['Accuracy']:.4f}")
    st.write(f"Precision: {metrics['Precision']:.4f}")
    st.write(f"Recall: {metrics['Recall']:.4f}")
    st.write(f"F1 Score: {metrics['F1']:.4f}")

st.write("Developed by Nora Bearth")