import os
import pandas as pd
import numpy as np
import shutil

# Define folder paths
data_folder = "data"
clean_folder = "data_clean"
not_used_folder = os.path.join(data_folder, "not_used")

# Ensure 'not_used' folder exists
os.makedirs(not_used_folder, exist_ok=True)

# Get the list of files in the data folder
data_files = set(f for f in os.listdir(data_folder) if f.endswith(".pkl"))
clean_files = set(os.listdir(clean_folder))  # Processed files

# List of new files to process (only those not in data_clean)
new_files = data_files - clean_files

# Loop over new files and process them individually
for file in new_files:
    file_path = os.path.join(data_folder, file)

    # Load the file into a DataFrame
    data = pd.read_pickle(file_path)

    # Check if 'name' column contains NaN values (or if any row has NaN in 'name')
    if data['name'].isna().any():
        # If 'name' contains NaN, move the file to the 'not_used' folder
        not_used_file_path = os.path.join(not_used_folder, file)
        shutil.move(file_path, not_used_file_path)
        print(f"Moved {file} to the 'not_used' folder due to NaN in 'name' column.")
        # Skip rest of the loop if not used
        continue

    if (~data.details_competition_type.isin(
        ["Men's Slalom", "Men's Giant Slalom", "Women's Slalom",
         "Women's Giant Slalom", "Men's Downhill", "Men's Super G",
         "Women's Super G", "Women's Downhill"])).any():

        # If the file doesn't match the competition types, move it to the 'not_used' folder
        not_used_file_path = os.path.join(not_used_folder, file)
        shutil.move(file_path, not_used_file_path)
        print(f"Moved {file} to the 'not_used' folder.")

    else:

        # Remove competition_type="Training"
        data = data[data.competition_type != "Training"]

        # Drop unneeded variables
        data.drop(['competition_type', 'start_altitude', 'finish_altitude',
                   'lenght', 'vertical_drop', 'number_of_gates_1',
                   'number_of_gates_2', 'start_time_2', 'run_1_time',
                   'run_2_time'], axis=1, inplace=True)

        # Rename variables and change formats
        data = data.rename(columns={'rank': 'total_rank'})
        data['date'] = pd.to_datetime(data.date, format='%B %d, %Y')
        data[['location', 'location_country']] = data.competition_location.str.split(" \(", expand=True)
        data.drop(['competition_location'], axis=1, inplace=True)
        data['location_country'] = data.location_country.str.replace('\)', '', regex=True)
        data['total_rank'] = pd.to_numeric(data['total_rank'])

        data['gender'] = data['details_competition_type'].str.contains("Men").map({True: 'men', False: 'women'})
        data['discipline'] = (data['details_competition_type'].str.extract(r'(Downhill|Slalom|Giant Slalom|Super G)').fillna('Other'))  


        # Create new variables
        data['own_trainer'] = np.where(data.course_setter_1_nation == data.country, 1, 0)
        data.loc[data.course_setter_2_nation == data.country, 'own_trainer'] = 1
        data['home_race'] = np.where(data.country == data.location_country, 1, 0)
        data['age'] = data['date'].dt.year - data['birthyear'].astype(int)

        # Get rank and attach world cup points
        wcp_dict = {1: 100, 2: 80, 3: 60, 4: 50, 5: 45, 6: 40, 7: 36, 8: 32,
                    9: 29, 10: 26, 11: 24, 12: 22, 13: 20, 14: 18, 15: 16,
                    16: 15, 17: 14, 18: 13, 19: 12, 20: 11, 21: 10, 22: 9,
                    23: 8, 24: 7, 25: 6, 26: 5, 27: 4, 28: 3, 29: 2, 30: 1}
        data['wcp'] = data.total_rank.map(wcp_dict)
        data['wcp'] = np.where(data.wcp.isna(), 0, data.wcp)

        # Create race indicator
        data["race_id"] = (data.discipline.astype(str) + '_' +
                           data.gender.astype(str) + '_' +
                           data.date.astype(str) + '_' +
                           data.location.astype(str))

        # Create dummies for race start times
        data['start_hour'] = data.start_time_1.fillna("0:00").str.split(':').str[0].astype(int)
        data['startmorning'] = (data.start_hour <= 11).astype(int)
        data['startafternoon'] = np.where((data.start_hour >= 12) & (data.start_hour <= 16), 1, 0)

        # Create dummies for the season period
        data['racemonth'] = data['date'].dt.month
        data['earlyseason'] = (data.racemonth.isin([10, 11])).astype(int)
        data['midseason'] = (data.racemonth.isin([12, 1, 2])).astype(int)

        # Create a variable for being on the podium or not
        data['podium'] = np.where(data.total_rank <= 3, 1, 0)

        # Create categories for strength of racer
        data["top20"] = 0
        data["top21_30"] = 0
        data.loc[(data.bib.astype(int) >= 1) & (data.bib.astype(int) <= 20), 'top20'] = 1 
        data.loc[(data.bib.astype(int) >= 21) & (data.bib.astype(int) <= 30), 'top21_30'] = 1

        # Create dummy for main nationalities

        # List of target countries
        top_countries = ['aut', 'sui', 'ita', 'usa', 'fra', 'ger', 'nor',
                         'swe', 'can', 'slo']
        # Create a dummy variable for each country
        for country in top_countries:
            data[country] = np.where(data['country'] == country, 1, 0)

        # Drop unneeded columns
        data.drop(['start_time_1', 'course_setter_1', 'course_setter_2', 'run',
                   'course_setter_1_nation', 'course_setter_2_nation', 'time',
                   'location_country', 'details_competition_type', 'birthyear'],
                  axis=1, inplace=True)

        # Save the cleaned data to the clean folder
        cleaned_file_path = os.path.join(clean_folder, file)
        data.to_pickle(cleaned_file_path)
        print(f"Processed and saved: {file}")
