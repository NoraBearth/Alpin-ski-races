import numpy as np
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

def get_attribute_find(row, html_class: str) -> str:
    try:
        # Extract the value
        value = row.find(
            'div', class_=html_class).text.strip()
    except AttributeError:
        value = np.nan
    return value


def get_attribute_find_all(row, html_class: str) -> str:
    try:
        # Extract the value
        value = row.find_all(
            'div', class_=html_class)[0].text.strip()
    except AttributeError:
        value = np.nan
    return value

def get_attribute_run_time(row, html_class_1: str, html_class_2: str,
                           column: int) -> str:
    try:
        # Extract the value
        value = row.find_all(
            'div', class_=html_class_1)[column].text.strip()
    except (AttributeError, IndexError, TypeError):
        
        try: 
            value = row.find(
                'div', class_=html_class_2)[column].text.strip()
        
        except (AttributeError, IndexError, TypeError): 
            value = np.nan
        
    return value

def get_element(driver, html_class: str) -> str:
    try:
        # Extract the value
        value = driver.find_element(By.CLASS_NAME, html_class)
        # Get the text content of the element
        value_text = value.text
    except NoSuchElementException:
        value_text = np.nan
    return value_text


        
        