fileopen = open(r"C:\Users\Anuj\Desktop\Kali\Data\Api.txt", "r")
API = fileopen.read()
fileopen.close()

import openai
from dotenv import load_dotenv

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def Brain(question, BLog = None):
    
    fileLog = open(r"C:\Users\Anuj\Desktop\Kali\Database\Chatlog.txt", "r")
    bConversation = fileLog.read()
    fileLog.close()
    
    if BLog == None:
        BLog = bConversation
    
    past_conversation = f"{BLog}You: {question}\nKali: "
    response = completion.create(
        model = "text-davinci-002",
        prompt = past_conversation,
        temperature = 0.5,
        max_tokens = 100,
        top_p = 0.3,
        frequency_penalty = 1,
    )    
    
    ans = response.choices[0].text.strip()
    updateConversation = bConversation + f"You: {question}\nKali: {ans}\n"
    fileLog = open(r"C:\Users\Anuj\Desktop\Kali\Database\Chatlog.txt", "w")
    fileLog.write(updateConversation)
    fileLog.close()
    return ans

