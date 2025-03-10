from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
from selenium.webdriver.edge.service import Service as EdgeService
from webdriver_manager.microsoft import EdgeChromiumDriverManager
import pandas as pd
import time as tm
from selenium.common.exceptions import NoSuchElementException
import numpy as np
import sys
import time
from datetime import datetime
from fis.functions_scraper import get_attribute_find, get_attribute_find_all, get_attribute_run_time, get_element
# %%

df = pd.read_pickle('data_clean/final_dataset.pkl')
latest_date = df['date'].max()
latest_date = latest_date.to_pydatetime()


# Define the starting and ending years
start_year = 2025
end_year = 2025  # Or use the current year dynamically

# Loop over all years
for year in range(start_year, end_year + 1):

    driver = webdriver.Edge(service=EdgeService(EdgeChromiumDriverManager().install()))

    url = f"https://www.fis-ski.com/DB/alpine-skiing/calendar-results.html?eventselection=results&place=&sectorcode=AL&seasoncode={year}&categorycode=WC&disciplinecode=&gendercode=&racedate=&racecodex=&nationcode=&seasonmonth=X-{year}&saveselection=-1&seasonselection="

    driver.get(url)

    # Close Cookie Consent Dialog if Present
    try:
        # Wait for the cookie consent dialog to appear and close it if found
        WebDriverWait(driver, 2).until(EC.element_to_be_clickable((By.XPATH, '//button[contains(text(), "Allow all")]'))).click()

    except Exception as e:
        print("No cookie consent dialog or failed to close:", e)

    # Get all the locations in a year
    locations = driver.find_elements(
        By.XPATH, '//a[@class="pr-1 g-lg-1 g-md-1 g-sm-2 hidden-xs justify-left"]')

    # Loop over all locations
    for location in range(len(locations)):

        for _ in range(3):  # Try 3 times
            locations = driver.find_elements(
                By.XPATH, '//a[@class="pr-1 g-lg-1 g-md-1 g-sm-2 hidden-xs justify-left"]')

            if locations:
                break  # Stop retrying if locations are found
            else:
                print("Locations not found, retrying in 2 seconds...")
                time.sleep(2)

        locations[location].click()
        tm.sleep(1)

        races = driver.find_elements(By.XPATH, '//a[@class="g-lg-2 g-md-3 g-sm-2 g-xs-4 px-md-1 px-lg-1 pl-xs-1 justify-left"]')

        for race in range(len(races)):

            page_source = driver.page_source
            soup = BeautifulSoup(page_source, 'html.parser')
            name_race_art = soup.find_all('div', class_='clip')

            date_text = soup.find_all('div', class_='timezone-date')

            dates = [div.get('data-date') for div in date_text]
            dates = [date for date in dates if date is not None]

            # Convert the date strings to datetime objects
            dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

            # Find the maximum date
            max_date = max(dates)
            
            if max_date < latest_date:

                # Stop the scraper entirely if a race is older or already scraped
                driver.quit()  # Close the browser
                print("Closing the scraper since no newer races are available.")
                sys.exit()  # Stop the script completely

            try:

                if len(name_race_art) > 2*race and name_race_art[2*race].get_text(strip=True) == 'Team Parallel':
                    print('Team event')

                else:
                    try:
                        races = driver.find_elements(
                            By.XPATH, '//a[@class="g-lg-2 g-md-3 g-sm-2 g-xs-4 px-md-1 px-lg-1 pl-xs-1 justify-left"]')
                        
                        while True:
                            try:
                                # Wait for the element to be clickable
                                race_element = WebDriverWait(driver, 5).until(EC.element_to_be_clickable(races[race]))
                                race_element.click()
                                break  # Exit loop after successful click

                            except Exception as e:
                                tm.sleep(2)

                        # Fetch the updated page source
                        soup = BeautifulSoup(driver.page_source, 'html.parser')

                        check_cancelled2 = soup.find_all('div', class_="event-status event-status_cancelled") 

                        if len(check_cancelled2) != 0:
                            print('Race cancelled_1')
                            driver.back()

                        else:

                            # Find the elements containing the text
                            competition_location = get_element(driver, "heading_l2")
                            competition_type = get_element(driver, "event-header__subtitle")
                            details_competition_type = get_element(driver, "event-header__kind")
                            date = get_element(driver, "date__full")
                            run = get_element(driver, "schedule-list__event")
                            time = get_element(driver, "time__value")

                            info = [competition_location, competition_type, 
                                    details_competition_type, date, run, time]

                            info = pd.DataFrame(
                                [info],
                                columns = ['competition_location', 'competition_type',
                                           'details_competition_type', 'date',
                                           'run', 'time'])
                            page_source = driver.page_source
                            soup = BeautifulSoup(page_source, 'html.parser')

                            # Find the section containing technical data
                            tech_data = soup.find_all('section', class_='section_more-info')
                            technical_details = pd.DataFrame(
                                np.nan, index=[0],
                                columns=['start_altitude', 'finish_altitude',
                                         'lenght', 'vertical_drop',
                                         'course_setter_1',
                                         'course_setter_1_nation',
                                         'course_setter_2',
                                         'course_setter_2_nation',
                                         'number_of_gates_1',
                                         'number_of_gates_2',
                                         'start_time_1', 'start_time_2'],
                                dtype='object')

                            for n in range(1, len(tech_data)-1):
                                # Find all the rows in the technical data table
                                rows = tech_data[n].find_all('div', class_='table-row')
                                
                                # Mapping for standard fields
                                field_mapping = {
                                    "Start Altitude": "start_altitude",
                                    "Finish Altitude": "finish_altitude",
                                    "Length": "length",
                                    "Vertical Drop": "vertical_drop",
                                    }

                                # Mapping for fields that depend on 'n'
                                multi_field_mapping = {
                                    "Course Setter": "course_setter",
                                    "Number of Gates": "number_of_gates",
                                    "Start Time": "start_time",
                                    }

                                # Loop through the rows and extract values
                                for row in rows:
                                    key_element = row.find('div', class_='g-xs-9 g-sm-7 g-md-10 g-lg-8 justify-left bold')
                                    if not key_element:
                                        continue  # Skip if key is missing

                                    key_text = key_element.get_text(strip=True)

                                    # Standard fields
                                    if key_text in field_mapping:
                                        technical_details.loc[0, field_mapping[key_text]] = row.find(
                                            'div', class_='justify-right'
                                            ).get_text(strip=True) if row.find('div', class_='justify-right') else np.nan

                                    # Fields that depend on `n`
                                    elif key_text in multi_field_mapping:
                                        column_name = f"{multi_field_mapping[key_text]}_{n}"
                                        value_element = row.find('div', class_='g-xs-11 g-sm-14 g-md-10 g-lg-13 justify-left')
        
                                        technical_details.loc[0, column_name] = value_element.get_text(strip=True) if value_element else np.nan

                                    # Special case: Course Setter (extract nationality)
                                    if key_text == "Course Setter":
                                        nation_element = row.find('span', class_="country__name-short")
                                        technical_details.loc[0, f"course_setter_{n}_nation"] = nation_element.get_text(strip=True) if nation_element else np.nan
                                          
                            # Find all the table rows (excluding the header row)
                            rows = soup.find_all('a', class_='table-row')
                            race_results = pd.DataFrame(np.nan, index=[0],
                                columns=['rank', 'bib', 'name', 'birthyear', 'country',
                                         'total_time', 'run_1_time', 'run_2_time'], dtype='object')
                            # Iterate over each row and extract the data
                            for i, row in enumerate(rows[1:]):

                                race_results.loc[i, ['rank']] = get_attribute_find(row, "g-lg-1 g-md-1 g-sm-1 g-xs-2 justify-right pr-1 bold")
                                race_results.loc[i, ['bib']] = get_attribute_find(row, "g-lg-1 g-md-1 g-sm-1 justify-center hidden-sm-down gray")

                                if start_year >= 2025:
                                    race_results.loc[i, ['name']] = get_attribute_find(row, "g-xs-24 justify-left athlete-name")

                                else:
                                    racer_name = np.nan
                                    for html_class in [
                                            "g-lg-14 g-md-14 g-sm-13 g-xs-11 justify-left bold", 
                                            "g-lg-10 g-md-10 g-sm-9 g-xs-8 justify-left bold",
                                            "g-lg-6 g-md-6 g-sm-5 g-xs-8 justify-left bold",
                                            "g-lg-4 g-md-4 g-sm-3 g-xs-8 justify-left bold",
                                            "g-lg-8 g-md-8 g-sm-7 g-xs-8 justify-left bold",
                                                        ]:
                                        try:
                                            racer_name = row.find('div', class_=html_class).text.strip()
                                        except AttributeError:
                                            # if attribute for racer name is not found, go to next class
                                            continue
                                        # if the racer name has been successfully found, we can stop
                                        break
                                    race_results.loc[i, ['name']] = racer_name

                                race_results.loc[i, ['birthyear']] = get_attribute_find_all(row, "g-lg-1 g-md-1 hidden-sm-down justify-left")
                                race_results.loc[i, ['country']] = get_attribute_find(row, "country__name-short")
                                race_results.loc[i, ['total_time']] = get_attribute_find(row, "g-lg-2 g-md-2 justify-right blue bold hidden-sm hidden-xs")
                                race_results.loc[i, ['run_1_time']] = get_attribute_run_time(
                                    row, "g-lg-2 g-md-2 g-sm-2 justify-right bold hidden-xs", 
                                    "g-lg-2 g-md-2 g-sm-2 justify-right bold hidden-xs", 0)
                                race_results.loc[i, ['run_2_time']] = get_attribute_run_time(
                                    row, "g-lg-2 g-md-2 g-sm-2 justify-right bold hidden-xs",
                                    "g-lg-2 g-md-2 g-sm-2 justify-right bold hidden-xs", 1)

                            driver.back()

                            # Put all the info together

                            df = pd.concat([pd.concat([technical_details]*len(race_results), ignore_index=True), race_results], axis=1)

                            df_fin = pd.concat([pd.concat([info]*len(race_results), ignore_index=True), df], axis=1)

                            date_obj = datetime.strptime(date, "%B %d, %Y")  # Convert to datetime object
                            formatted_date = date_obj.strftime("%Y-%m-%d")

                            df_fin.to_pickle(f"data//{formatted_date}_{details_competition_type}.pkl")

                    except NoSuchElementException:
                        print('Race cancelled_2')
                        driver.back()
            except IndexError as e:
                print(f"IndexError at race {race}: {e}")

            tm.sleep(1)
        driver.back()
        tm.sleep(1)

    driver.quit()
