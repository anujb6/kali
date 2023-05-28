import pyttsx3

#windows based voice assistant
#advantages: Fast, Offline
#disadvantages: Robotic voice, Not so good for long conversations, less voices
def WindowSpeak(text):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    engine.setProperty('rate', 170)
    print("")
    print(f"you: {text}")
    print("")
    engine.say(text)
    engine.runAndWait()
    
#chorme based voice assistant
#advantages: more voices, more clear, overspeaking
#disadvantages: slow, requires internet, word limit
from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from time import sleep
# from webdriver_manager.chrome import ChromeDriverManager
# from selenium.webdriver.chrome.service import Service

chromeOptions = Options()
chromeOptions.binary_location = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
chromeOptions.add_argument('--headless')
chromeOptions.add_argument("--log-level=3") 
#it is deprecated but we dont have to install chrome driver everytime, if in future it does not work then we use above code
# driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chromeOptions)
driver = webdriver.Chrome(r"Database\chromedriver.exe", options=chromeOptions)
driver.maximize_window()

driver.get(r"https://ttsmp3.com/text-to-speech/British%20English/") 
ButtonSelecion = Select(driver.find_element(By.XPATH, value='/html/body/div[4]/div[2]/form/select'))
ButtonSelecion.select_by_visible_text('US English / Matthew')

def ChromeSpeak(text):
    length = len(str(text))
    if length == 0:
        pass
    else:
        print("")
        print(f"AI: {text}")
        print("")
        Data = str(text)
        driver.find_element(By.XPATH,
                            value='/html/body/div[4]/div[2]/form/textarea').send_keys(Data)
        driver.find_element(By.XPATH,
                            value='//*[@id="vorlesenbutton"]').click()
        driver.find_element(By.XPATH,
                            value='/html/body/div[4]/div[2]/form/textarea').clear()
        
        if length >= 30:
            sleep(4)
        elif length >= 40:
            sleep(6)
        elif length >= 55:
            sleep(8)
        elif length >= 70:
            sleep(10)
        elif length >= 100:
            sleep(13)                
        elif length > 100:
            sleep(14)
        else:
            sleep(3)
