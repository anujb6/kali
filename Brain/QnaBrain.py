fileopen = open(r"C:\Users\Anuj\Desktop\Kali\Data\Api.txt", "r")
API = fileopen.read()
fileopen.close()

import openai
from dotenv import load_dotenv

openai.api_key = API
load_dotenv()
completion = openai.Completion()

def QuestionAnswerBrain(questions, qnaLog = None):
    
    fileLog = open(r"C:\Users\Anuj\Desktop\Kali\Database\Qnalog.txt", "r")
    qnaConversation = fileLog.read()
    fileLog.close()
    
    if qnaLog == None:
        qnaLog = qnaConversation
    
    past_conversation = f"{qnaLog}Question: {questions}\nAnswer: "
    response = completion.create(
        model = "text-davinci-002",
        prompt = past_conversation,
        temperature = 0,
        max_tokens = 100,
        top_p = 1,
        frequency_penalty = 0,
    )    
    
    answer = response.choices[0].text.strip()
    updateConversation = qnaConversation + f"You: {questions}\n Answer: {answer}\n"
    fileLog = open(r"C:\Users\Anuj\Desktop\Kali\Database\Qnalog.txt", "w")
    fileLog.write(updateConversation)
    fileLog.close()
    return answer

