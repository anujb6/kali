from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os
from time import sleep
import pandas as pd 
from Body.Vocal import ChromeSpeak as Speak
from Body.Hear import Relating as MicExecution
import pathlib
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

scriptDirectory = pathlib.Path().absolute()

chromeOptions = Options()
chromeOptions.add_argument('--headless')
chromeOptions.add_experimental_option("excludeSwitches", ["enable-logging"])
chromeOptions.add_argument("--profile-directory=Default")
chromeOptions.add_argument(f"--user-data-dir={scriptDirectory}\\User Data")
driver = webdriver.Chrome(r"Database\chromedriver.exe", options=chromeOptions)
driver.maximize_window()
driver.get("https://web.whatsapp.com/")

os.system("")
os.environ["WDM_LOG_LEVEL"] = "0"

ListWeb = {'high court' : "+919833004751",
            'supreme court': "+919322254209",
            "test": '+919833190005'}

def SendMessage(Name):
    Speak(f"Preparing To Send a Message To {Name}")
    Speak("What's The Message By The Way?")
    Message = MicExecution()
    Number = ListWeb[Name.lower()]
    print(Number, " :this is the number")
    LinkWeb = 'https://web.whatsapp.com/send?phone=' + Number + "&text=" + Message
    driver.get(LinkWeb)
    print(LinkWeb, " :this is the link")
    element_present = EC.presence_of_element_located((By.XPATH, "/html/body/div[1]/div/div/div[5]/div/footer/div[1]/div/span[2]/div/div[2]/div[2]/button"))
    WebDriverWait(driver, 10).until(element_present)
    try:
        driver.find_element(By.XPATH, value='/html/body/div[1]/div/div/div[5]/div/footer/div[1]/div/span[2]/div/div[2]/div[2]/button').click()
        Speak("Message Sent")
        return True
    except:
        print("Due to Techincal Error or wrong contact, Message Not Sent")
        return False