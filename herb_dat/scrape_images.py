# Goal: Scrape a bunch of species from the CNH site
# Specifically using the Boston Harbor Islands National Recreation Area

# imports 
from selenium import webdriver
import json
import os
from pprint import pprint
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException
import time

chromedriver_location = os.getcwd() + '/chromedriver'
driver = webdriver.Chrome(chromedriver_location)

CNH_SITE = "https://neherbaria.org/portal/checklists/checklist.php?clid=28&pid=3"
CNH_CHECKLIST = "https://neherbaria.org/portal/checklists/"

driver.implicitly_wait(10)
driver.get(CNH_SITE)

taxalist = driver.find_element_by_id("taxalist-div")

main_window = driver.current_window_handle

family_and_species_divs = taxalist.find_elements_by_tag_name("div")
species_info = {}

current_family = ""
# current_family = "POACEAE"
# found = False
for div in family_and_species_divs:
    # found |= div.text == "Phragmites australis"

    # if not found:
    #     continue

    entry_type = div.get_attribute("class")

    if entry_type == "family-div":
        current_family = div.text

    if len(div.text.split(" ")) < 2:
        continue

    else:
        a_tag = div.find_element_by_tag_name("a")

        # Click on link for current species
        a_tag.click()
        # Make sure that the focus is the new window
        driver.switch_to_window(driver.window_handles[-1])

        # Click on button to show first 100 images and wait a little bit
        try:
            expand_images_btn = driver.find_element_by_xpath(
                '//a[@onclick="expandExtraImages();return false;"]')
            expand_images_btn.click()
            time.sleep(1)
        except:
            driver.close()
            time.sleep(0.7)
            driver.switch_to_window(main_window)
            continue
        

        # Find divs that wrap up each image thumbnail so we can go to the image site for said species
        species_images_list = driver.find_elements_by_class_name("tptnimg")

        images = []
        # Iterate through different pictures!
        for thumbnail in species_images_list:
            # Open current image page from list of images in new tab
            image_page = thumbnail.find_element_by_tag_name("a")
            driver.execute_script(
                "arguments[0].target='_blank';", image_page)
            image_page.click()
            # Make sure that the focus is the new window
            driver.switch_to_window(driver.window_handles[-1])
            
            try:
                # Open image up
                a_to_image_file = driver.find_element_by_xpath('//a[contains(text(), "Open")]')
                a_to_image_file.click()
                driver.switch_to_window(driver.window_handles[-1])
                
                src = driver.page_source
                if "503 Service Unavailable" not in src:
                    img = driver.find_element_by_tag_name("img")
                    species_image = img.get_attribute("src")
                else:
                    species_image = False
            except NoSuchElementException:
                species_image = False

            driver.close()
            time.sleep(0.7)
            driver.switch_to_window(driver.window_handles[-1])

            if species_image:
                images.append(species_image)
            
            # Close new tabs and switch to original
            driver.close()
            time.sleep(0.7)
            driver.switch_to_window(driver.window_handles[-1])

            if len(images) > 10:
                break

        driver.close()
        time.sleep(0.7)
        driver.switch_to_window(main_window)

        species_info[div.text] = {
            "family": current_family,
            "name": div.text,
            "images": images
        }

        pprint(species_info[div.text])
    
    with open("bruh.json", "w+") as outfile:
        json.dump(species_info, outfile)



driver.quit()
