from Brain.Brain import Brain
from Brain.QnaBrain import QuestionAnswerBrain 
from Body.Hear import Relating as Mic
print(">> initiating kali: wait for some seconds...")
from Body.Vocal import ChromeSpeak as Speak
from Features.Clap import Tester as Clap
print(">> initiating kali: wait for Few seconds...")
from Integrate import TaskInitializer as Task

def Main():
    Speak("Hello Friend!")
    while True:
        query = str(Mic()).replace(".", "")
        # query = str(input("You: "))
        valueReturn = Task(query)
        if valueReturn == True:
            pass
        elif len(query) < 3:
            pass
        elif type(valueReturn) == list:
            Speak(valueReturn[0])
        elif query == "False":
            pass
        elif any(x in query.split() for x in ["what is", "where is", "question", "answer", "who is"]):
            Reply = QuestionAnswerBrain(query)
        else:
            Speak(Brain(query))

def ClapDetect():
    query = Clap()
    if "True-mic" in query:
        print("")
        print(">> Clap Detected! <<")
        print("")   
        Main()
    else:
        pass
                
ClapDetect()            