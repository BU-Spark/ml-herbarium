# Used to fetch database of different plant names to use for prediction validation/lookup table
# Uses POWO site, whcih powers 

# imports 
from selenium import webdriver
import os
from selenium.webdriver.support.ui import Select
import pickle
import time

# Perform an empty search to be able to get all 1,191,241 resultss
URL = "http://powo.science.kew.org/?q="
chromedriver_location = os.getcwd() + '/chromedriver'
driver = webdriver.Chrome(chromedriver_location)

# Get the page
driver.implicitly_wait(10)
driver.get(URL)
# Use set to store all species names
species_names = set()

# Make sure only species show up
show_species = driver.find_element_by_class_name("facet.species_f ")
show_species.click()
time.sleep(1)

# Show 480 species per page
select_480_items = Select(driver.find_element_by_class_name("c-per-page.form-control"))
select_480_items.select_by_value("480")
time.sleep(1)

# Next button will be used to go to the next page if possible
next_button = driver.find_element_by_id("paginate-next")

# Iterate through all the species pages until you've reached the end
i = 0
j = 0
while True:
    print(i)
    # Find all species cards
    card_titles = driver.find_elements_by_class_name("c-card__title")

    # Add all new species names to set
    new_species_names = set(h2.find_element_by_tag_name("em").text for h2 in card_titles)
    # print(new_species_names)

    species_names.update(new_species_names)

    if next_button.is_enabled():
        query = next_button.find_element_by_tag_name("a").get_attribute("href")
        driver.get(query)
        next_button = driver.find_element_by_id("paginate-next")
    else:
        break

    time.sleep(1)

    if i % 100 == 0 or i == 2041:
        with open('species_names%i.txt' % (j), 'wb+') as f:
            pickle.dump(species_names, f)
        j += 1
        species_names = set()
    i += 1
# paginate-next
