import os 
import keyboard
import pyautogui
import webbrowser
from time import sleep

def OpenExe(Query):
    Query = str(Query).lower()
    
    if any(x in Query.split() for x in ["visit", "launch"]):
        NameofWebsite =  Query.replace("visit ","") if "visit" in Query else Query.replace("launch ","")
        link = f"https://www.{NameofWebsite}.com"
        webbrowser.open(link)
    
    elif any(x in Query.split() for x in ["open", "start"]):
        NameOfTheApp = Query.replace("open ","") if "open" in Query else Query.replace("start ","")
        pyautogui.press("win")
        sleep(0.5)
        keyboard.write(NameOfTheApp)
        sleep(0.5)
        keyboard.press('enter')
        sleep(0.5)
        return True
       