import os
import pandas as pd

# %% Data loading

# Define folder paths
clean_folder = "data_clean"

# Get the list of files in both folders
data_files = set(f for f in os.listdir(clean_folder) if f.endswith(".pkl"))

# Create an empty list to store DataFrames
data_frames = []

# Process only new files
for file in data_files:
    file_path = os.path.join(clean_folder, file)

    # Load the file into a DataFrame
    df = pd.read_pickle(file_path)

    # Append the DataFrame to the list
    data_frames.append(df)

# Concatenate all DataFrames into one
data = pd.concat(data_frames, ignore_index=True)

# Drop duplicates
data.drop_duplicates(subset=['race_id', 'name'], inplace=True)

# %% Feature creation over different races

# Sort data by athlete name and race date
data = data.sort_values(['name', 'discipline', 'date'])
# Compute rolling average of WCP points for last 10, 5, and 1 races
data['wcp_last_10'] = data.groupby(['name', 'discipline'])['wcp'].transform(lambda x: x.rolling(10, min_periods=1).mean())
data['wcp_last_5'] = data.groupby(['name', 'discipline'])['wcp'].transform(lambda x: x.rolling(5, min_periods=1).mean())
data['wcp_last_1'] = data.groupby(['name', 'discipline'])['wcp'].shift(1).fillna(0)  # Last race's WCP points
data['team_wcp_last_10'] = data.groupby(['country', 'discipline'])['wcp'].transform(lambda x: x.rolling(10, min_periods=1).mean())
data['team_wcp_last_5'] = data.groupby(['country', 'discipline'])['wcp'].transform(lambda x: x.rolling(5, min_periods=1).mean())
data['team_wcp_last_1'] = data.groupby(['country', 'discipline'])['wcp'].shift(1).fillna(0)  # Last race's WCP points

# %% Save dataset

# Save the cleaned data to the clean folder
cleaned_file_path = os.path.join(clean_folder, 'final_dataset.pkl')
data.to_pickle(cleaned_file_path)
