from googletrans import Translator
import speech_recognition as sr

# listening to the person
def Hear():
    ear = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        ear.pause_threshold = 1
        audio = ear.listen(source, 0, 8)
    try:
        print("Recognizing...")
        query = ear.recognize_google(audio, language='en-IN')
    except:
        return False
    
    query = str(query).lower()
    return query

def Translate(text):
    translate = Translator()
    result = translate.translate(str(text), dest='en')
    return result.text

def Relating():
    query = Hear()
    if query == False:
        return False
    else:
        return query

#we are not using translate function for now    
 
