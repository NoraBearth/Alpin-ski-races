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
selected_tab = st.sidebar.radio("Go to:", ["General Information", "Descriptive Statistics", "Predictions & Actual Results"])

if selected_tab == "General Information":
    st.header("Welcome to the Alpine ski race Dashboard!")
    st.write(
        """
        This dashboard allows you to explore descriptive statistics, predictions and
        actual results for different races of the Alpine Ski World cup.

        ### How the machine learning models were trained:

        The models have been trained on race data, with various features such
        as the athletes' statistics, race history, and personal details.
        The models predict the probability of a racer finishing in the top 3
        positions (on the podium) for each race.

        The data was split into **training** and **testing** sets based on the
        race season. The training data consists of early seasons, and the
        testing data includes races of the last two seasons.

        Each model estimates the probability that a racer will be on the podium
        for a given race. The output of the models is a **probability score**
        for each athlete, where a higher probability means a higher likelihood
        of finishing in the top 3 positions.

        The dashboard was only done for fun.

        """
    )

elif selected_tab == "Descriptive Statistics":
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
    
    # Sort the top_males and top_females dataframes by "wcp" in descending order
    top_males_sorted = top_males.sort_values(by='wcp', ascending=False)
    top_females_sorted = top_females.sort_values(by='wcp', ascending=False)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Male Athletes")
        st.plotly_chart(px.bar(top_males_sorted, x="wcp", y="name", orientation="h", title=f"Top 10 Males {season}"))
    with col2:
        st.subheader("Top 10 Female Athletes")
        st.plotly_chart(px.bar(top_females_sorted, x="wcp", y="name", orientation="h", title=f"Top 10 Females {season}"))

    # Athlete performance graph
    athlete = st.selectbox("Select an athlete", df["name"].unique())
    if athlete:
        athlete_df = df[df["name"] == athlete]
        yearly_points = athlete_df.groupby("season")["wcp"].sum().reset_index()
        st.plotly_chart(px.line(yearly_points, x="season", y="wcp", title=f"{athlete}'s World Cup Points"))

elif selected_tab == "Predictions & Actual Results":
    st.header("Predictions & Actual Results")
    
    # Race and model selection
    selected_race = st.selectbox("Select Race", df['race_id'].unique())
    selected_model = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Lasso"])

    # Default figure and message when no race is selected
    default_figure = px.line(title="Please select a race to make a prediction.")
    default_message = "Select a race and model to see predictions and actual results."

    if selected_race:
        # Filter data for 2024 and beyond
        df_2024 = df[df.season >= 2024].copy()

        # Get the race data for the selected race
        race_data = df_2024[df_2024['race_id'] == selected_race].copy()

        # Feature selection
        X_NAME_cat = ['gender', 'discipline']
        X_NAME_numeric = ['age', 'wcp_last_10', 'wcp_last_5', 'wcp_last_1',
                      'team_wcp_last_10', 'team_wcp_last_5', 'team_wcp_last_1']
        X_NAME_binary = ['own_trainer', 'home_race', 'startmorning', 'startafternoon',
                     'earlyseason', 'midseason', 'top20', 'top21_30',
                     'aut', 'sui', 'ita', 'usa', 'fra', 'ger', 'nor', 'swe',
                     'can', 'slo']
        X_NAME = X_NAME_numeric + X_NAME_cat + X_NAME_binary

        feature_df = race_data[X_NAME].copy()

        # Load the selected model
        model_map = {
        "Random Forest": 'models/model_random_forest.pkl',
        "Gradient Boosting": 'models/model_gradient_boosting.pkl',
        "Lasso": 'models/model_lasso.pkl'
        }

        model = joblib.load(model_map.get(selected_model, ''))
        if not model:
            st.write("Invalid model selected.")
            st.plotly_chart(default_figure)
            st.write(default_message)
            st.stop()

        # Make predictions
        prediction = model.predict_proba(feature_df)[:, 1].copy()
        race_data['Predicted'] = prediction

        predictions = race_data.sort_values(by=['race_id', 'Predicted'], ascending=[True, False])
        predictions['predicted_rank'] = predictions.groupby('race_id')['Predicted'].rank(method="first", ascending=False)
        predictions['predicted_podium'] = predictions['predicted_rank'].apply(lambda x: 1 if x <= 3 else 0)
    
        top_3_athletes = predictions[predictions['predicted_podium'] == 1].copy()
        top_3_athletes['predicted_rank'] = top_3_athletes['predicted_rank'].astype(int).astype('category')

        # Prediction Plot
        fig_pred = px.bar(
            top_3_athletes,
            x='name',
            y='Predicted',
            color='predicted_rank',
            title=f"Top 3 Predicted Athletes for Race: {selected_race}",
            labels={"name": "Athlete", "Predicted": "Prediction Probability"},
            category_orders={"predicted_rank": [1, 2, 3]},
            color_discrete_sequence=["#FFD700", "#C0C0C0", "#CD7F32"],
            text_auto=True,
            )

        fig_pred.update_layout(
            legend_title="Predicted Rank",
            yaxis_title="Prediction Probability",
            xaxis_title="Athlete",
            plot_bgcolor="white",
            margin=dict(t=50, b=50, l=50, r=50),
            barmode='group'
            )

        # Display Prediction Plot in Streamlit
        st.subheader(f"Predicted Podium for Race: {selected_race}")
        st.plotly_chart(fig_pred)

        # Actual Results
        actual_results = df[(df['race_id'] == selected_race) & (df['total_rank'] <= 3)]  
        actual_results = actual_results.sort_values(by='total_rank')
        actual_results["bar_height"] = 4 - actual_results["total_rank"]  # 1 -> 3, 2 -> 2, 3 -> 1
        actual_results['total_rank'] = actual_results['total_rank'].astype(int).astype('category')
        actual_results['bar_height'] = actual_results['bar_height'].astype(int).astype('category')

        fig_truth = px.bar(
            actual_results,
            x="name",  
            y="bar_height",  # Use the inverted ranking for correct bar heights
            color="total_rank",
            title="Actual Podium Finishers for Race",
            labels={"name": "Athlete", "bar_height": "Podium Position"},
            category_orders={"total_rank": [1, 2, 3]},  
            color_discrete_sequence=["#FFD700", "#C0C0C0", "#CD7F32"],  # Gold, Silver, Bronze colors
            text_auto=True)

        # Adjust axis settings to correctly reflect podium ranking
        fig_truth.update_layout(
            yaxis_title="Podium Position (1st on top)",  
            xaxis_title="Athlete",
            yaxis=dict(tickvals=[1, 2, 3], ticktext=["3rd", "2nd", "1st"]),  # Fix tick labels
            plot_bgcolor="white",
            margin=dict(t=50, b=50, l=50, r=50))

        # Display Actual Results in Streamlit
        st.subheader("Actual Podium")
        st.plotly_chart(fig_truth)

        # Metrics Calculation
        df_test = df[df['date'] >= pd.Timestamp('2024-03-09')].copy()
        feature_df = df_test[X_NAME].copy()
        predictions = model.predict_proba(feature_df)[:, 1].copy()
        df_test['Predicted'] = predictions
        df_test = df_test.sort_values(by=['race_id', 'Predicted'], ascending=[True, False])

        df_test['predicted_rank'] = df_test.groupby('race_id')['Predicted'].rank(method="first", ascending=False)
        df_test['predicted_podium'] = df_test['predicted_rank'].apply(lambda x: 1 if x <= 3 else 0)

        # Calculate metrics for all races
        overall_metrics = evaluation(df_test['podium'], df_test['predicted_podium'])

        model_names = {"Random Forest": "Random Forest",
                   "Gradient Boosting": "Gradient Boosting",
                   "Lasso": "Lasso"}

        # Display Model Evaluation Metrics
        model_full_name = model_names.get(selected_model, selected_model)
        st.subheader("Model Evaluation Metrics")
        st.write(f"Model: {model_full_name}")
        st.write(f"Accuracy: {overall_metrics['Accuracy']:.4f}")
        st.write(f"Precision: {overall_metrics['Precision']:.4f}")
        st.write(f"Recall: {overall_metrics['Recall']:.4f}")
        st.write(f"F1 Score: {overall_metrics['F1']:.4f}")

        # Display Note
        st.write(
            "Note: Precision and Recall, and therefore also the F1 score, are the same because the model "
            "is predicting the top 3 racers as 'on the podium' based on the predicted probabilities. "
            "Since only the top 3 are selected, a wrong prediction (false positive) on one of the top 3 "
            "results in a corresponding missed prediction (false negative) for a racer who should have been on the podium. "
            "This balance between false positives and false negatives causes the Precision and Recall to be identical."
            )

    else:
        # Show default view when no race is selected
        st.plotly_chart(default_figure)
        st.write(default_message)

    st.write("Developed by Nora Bearth")