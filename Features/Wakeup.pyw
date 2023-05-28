import speech_recognition as sr
import os


def Listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        r.pause_threshold = 1
        audio = r.listen(source, 0, 8)
        
        try:
            print("Recognizing...")
            query = r.recognize_google(audio, language='en-in')
        except:
            return ""
    query = str(query).lower()
    print(query)
    return query    


def Wakeup():
    query = Listen().lower()
    if "wake up kali" in query:
        return os.startfile(r"C:\Users\Anuj\Desktop\Kali\Kali.py")
    elif "close kali" in query:
        return os.close(r"C:\Users\Anuj\Desktop\Kali\Kali.py")  
    else:
        pass
    
   