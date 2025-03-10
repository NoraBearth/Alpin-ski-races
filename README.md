# Ski-race-predictions

## Overview

The goal of this repository is to build an interactive dashboard that displays various descriptive statistics and predictions for the podium positions of Alpine Ski World Cup races. Data from the FIS website was scraped and different machine learning models were trained to predict the probability of a racer finishing in the top three positions.

The dashboard allows users to view predictions and actual results for the races of this season, providing valuable insights into how well the models perform. The final application is deployed on [Streamlit Cloud](https://alpin-ski-races.streamlit.app/), where it is accessible to the public.

## Features

- **Race Prediction**: Predict the top 3 finishers for upcoming races based on historical data.
- **Model Evaluation**: Evaluate model performance using metrics such as accuracy, precision, recall, and F1 score.
- **Descriptive Statistics**: Visualize various statistics about the athletes, such as top-ranked athletes and historical performance.
- **Interactive Dashboard**: Easily interact with race data and model results through a simple, user-friendly interface built with Streamlit.

## Machine Learning Models

The project leverages several machine learning models to predict the top three finishers of each race:

1. **Random Forest**
2. **Gradient Boosting**
3. **Lasso**

These models were trained on data from previous seasons to predict the likelihood of athletes finishing on the podium (top three) for upcoming races.

## Data

The data for this project was scraped from the [FIS website](https://www.fis-ski.com/) and includes various features such as:

- **Athlete Information**: Gender, discipline, and personal statistics.
- **Race Information**: Historical race performance and athlete rankings.
- **World Cup Points**: Performance data from previous races to make accurate predictions.

## How It Works

1. **Data Scraping**: The FIS website is scraped to collect data on past races, athletes, and performance metrics.
2. **Data Cleaning and Feature Creation**: The data is cleaned and different features are created.
3. **Model Training**: Historical race data is used to train the machine learning models (Random Forest, Gradient Boosting, and Lasso).
4. **Race Predictions**: The trained models predict the probability of each athlete finishing in the top three positions for upcoming races.
5. **Dashboard**: An interactive Streamlit dashboard displays descriptive statistics, race predictions, actual race results, and model evaluation metrics.

## Deployment

The application is deployed on [Streamlit Cloud](https://alpin-ski-races.streamlit.app/). You can interact with the dashboard live by visiting the link.

## Disclaimer

This repository was created for fun and educational purposes. It is not intended for commercial use. Enjoy the dashboard and have fun! 😄
