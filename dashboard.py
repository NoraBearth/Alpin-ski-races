import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import webbrowser
import joblib
from fis.evaluation import evaluation
from fis import dashboard_functions
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Import data
df = pd.read_pickle('data_clean/final_dataset.pkl')

# Convert 'date' column to datetime
df["date"] = pd.to_datetime(df["date"])


# Convert 'date' column to datetime
df["date"] = pd.to_datetime(df["date"])

# Extract year from date
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df['season'] = df.apply(lambda row: row['year'] if row['month'] >= 10 else row['year'] - 1, axis=1)
df_2025_and_beyond = df[df.season >= 2024].copy()

app = dashboard_functions.run_where(place='local')

# Create a list of available race names and their respective race IDs for the dropdown
race_options = [{"label": race_id, "value": race_id} for race_id in df_2025_and_beyond['race_id'].unique()]

app.layout = html.Div([
    html.H1("Ski World Cup Dashboard", style={"text-align": "center"}),

    dcc.Tabs(id="tabs", value="tab1", children=[
        dcc.Tab(label="Descriptive Statistics", value="tab1"),
        dcc.Tab(label="Predictions & Actual Results", value="tab2"),
    ]),

    html.Div(id="tabs-content")  # Placeholder for tab content
])


@app.callback(
    Output("tabs-content", "children"),
    [Input("tabs", "value")]
)
def render_tab_content(selected_tab):
    if selected_tab == "tab1":
        return html.Div([
            html.H2("Descriptive Statistics"),

            # Dropdown to select a season
            dcc.Dropdown(
                id="season-selector",
                options=[{"label": str(y), "value": y} for y in sorted(df["season"].unique())],
                value=df["season"].max(),
                clearable=False,
                style={"width": "50%"}
            ),

            # Leaderboard graphs
            dcc.Graph(id="male-wcp-leaderboard"),
            dcc.Graph(id="female-wcp-leaderboard"),

            # Athlete performance dropdown + graph
            dcc.Dropdown(
                id="athlete-selector",
                options=[{"label": name, "value": name} for name in df["name"].unique()],
                placeholder="Select an athlete",
                style={"width": "50%"}
            ),
            dcc.Graph(id="athlete-performance"),
        ])

    elif selected_tab == "tab2":
        return html.Div([
            html.H2("Predictions & Actual Results"),

            # Race & Model Selector
            dcc.Dropdown(
                id="race-selector",
                options=race_options,
                value=None,
                placeholder="Select a race",
                clearable=False,
                style={"width": "50%"}
            ),

            dcc.Dropdown(
                id="model-selector",
                options=[
                    {"label": "Random Forest", "value": "rf"},
                    {"label": "Gradient Boosting", "value": "gb"},
                    {"label": "Lasso", "value": "lasso"}
                ],
                value="rf",
                clearable=False,
                style={"width": "50%"}
            ),

            # Prediction & Truth Graphs & metrics
            dcc.Graph(id="prediction-graph"),
            dcc.Graph(id="truth-graph"),
            html.Div(id="metrics-display"),
        ])

    return html.Div("Select a tab to display content.")


# Callbacks
@app.callback(
    [Output("male-wcp-leaderboard", "figure"), Output("female-wcp-leaderboard", "figure")],
    [Input("season-selector", "value")]
)
def update_wcp_leaderboard(selected_season):
    # Filter data for the selected year
    filtered_df = df[df["season"] == selected_season].copy()

    # Separate by gender
    male_df = filtered_df[filtered_df["gender"] == "men"].groupby("name", as_index=False)["wcp"].sum()
    female_df = filtered_df[filtered_df["gender"] == "women"].groupby("name", as_index=False)["wcp"].sum()

    # Get top 10 athletes for each gender
    top_males = male_df.nlargest(10, "wcp")
    top_females = female_df.nlargest(10, "wcp")

    # Create bar plots
    fig_male = px.bar(
        top_males,
        x="wcp",
        y="name",
        orientation="h",
        title=f"Top 10 Male Athletes by World Cup Points ({selected_season})",
        labels={"wcp": "World Cup Points", "name": "Athlete"},
    )

    fig_female = px.bar(
        top_females,
        x="wcp",
        y="name",
        orientation="h",
        title=f"Top 10 Female Athletes by World Cup Points ({selected_season})",
        labels={"wcp": "World Cup Points", "name": "Athlete"},
    )

    return fig_male, fig_female


@app.callback(
    dash.Output("athlete-performance", "figure"),
    [dash.Input("athlete-selector", "value")]
)
def update_athlete_performance(selected_athlete):
    if not selected_athlete:
        return px.line(title="Select an athlete to view performance")

    athlete_df = df[df["name"] == selected_athlete]

    # Ensure the year range from 2015 to 2024, fill missing years with zero.
    all_years = pd.DataFrame({'season': range(2015, 2025)})
    yearly_points = athlete_df.groupby("season")["wcp"].sum().reset_index()

    # Merge to ensure all years are represented, filling missing years with zero
    yearly_points = pd.merge(all_years, yearly_points, on="season", how="left").fillna(0)

    fig = px.line(
        yearly_points,
        x="season",
        y="wcp",
        title=f"{selected_athlete}'s Total World Cup Points Per Season",
        labels={"wcp": "Total World Cup Points", "season": "Season"},
        markers=True)

    # Set ticks to show only full years from 2015 to 2024
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(2015, 2025))))
    return fig


@app.callback(
    [dash.Output("prediction-graph", "figure"),
     dash.Output("truth-graph", "figure"),
     dash.Output("metrics-display", "children")],
    [dash.Input("race-selector", "value"),
     dash.Input("model-selector", "value")]
)
def update_predictions_and_truth(selected_race, selected_model):
    # Default figure and message when no race is selected
    default_figure = px.line(title="Please select a race to make a prediction.")
    default_message = "Select a race and model to see predictions and actual results."

    if not selected_race:
        # Return the default figures and message when no race is selected
        return default_figure, default_figure, default_message

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
    if selected_model == "rf":
        model = joblib.load('models/model_random_forest.pkl')
    elif selected_model == "gb":
        model = joblib.load('models/model_gradient_boosting.pkl')
    elif selected_model == "lasso":
        model = joblib.load('models/model_lasso.pkl')
    else:
        return default_figure, default_figure, default_message

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

    # **Actual Results**
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

    model_names = {"rf": "Random Forest",
                   "gb": "Gradient Boosting",
                   "lasso": "Lasso"}

    # Use this to get the full name based on the selected model
    model_full_name = model_names.get(selected_model, selected_model)

    # Create a display table for the overall metrics
    metrics_display = html.Div([
        html.H3(f"Overall Model Evaluation Metrics: {model_full_name}"),
        html.Table([
            html.Tr([html.Th("Metric"), html.Th("Value")]),
            html.Tr([html.Td("Accuracy"), html.Td(f"{overall_metrics['Accuracy']:.4f}")]),
            html.Tr([html.Td("Precision"), html.Td(f"{overall_metrics['Precision']:.4f}")]),
            html.Tr([html.Td("Recall"), html.Td(f"{overall_metrics['Recall']:.4f}")]),
            html.Tr([html.Td("F1 Score"), html.Td(f"{overall_metrics['F1']:.4f}")]),
            ]),
        # Add a note explaining why Precision and Recall are the same
        html.Div([
            html.P(
            "Note: Precision and Recall, and therefore also the F1 score, are the same because the model "
            "is predicting the top 3 racers as 'on the podium' based on the predicted probabilities. "
            "Since only the top 3 are selected, a wrong prediction (false positive) on one of the top 3 "
            "results in a corresponding missed prediction (false negative) for a racer who should have been on the podium. "
            "This balance between false positives and false negatives causes the Precision and Recall to be identical."
            )
            ])
        ])


    return fig_pred, fig_truth, metrics_display

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:8050/")
    app.run_server(debug=True)

# if __name__ == "__main__":
#     app.run_server(host="0.0.0.0", port=8050)
