import nltk 
from nltk.stem.porter import PorterStemmer
import torch
import torch.nn as nn
import json
import numpy as np 
import random
from torch.utils.data import Dataset,DataLoader
from Features.Open import OpenExe
from Features.Whatsapp import SendMessage
from Features.Nasa import Apod_Nasa as Nasa
from Body.Vocal import ChromeSpeak as Speak
nltk.download('punkt')

def TrainTasks():

    class NeuralNet(nn.Module):

        def __init__(self,input_size,hidden_size,num_classes):
            super(NeuralNet,self).__init__()
            self.l1 = nn.Linear(input_size,hidden_size)
            self.l2 = nn.Linear(hidden_size,hidden_size)
            self.l3 = nn.Linear(hidden_size,num_classes)
            self.relu = nn.ReLU()

        def forward(self,x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out

    stemmer = PorterStemmer()

    def tokenize(sentence):
        return nltk.word_tokenize(sentence)

    def stem(word):
        return stemmer.stem(word.lower())

    def bagOfWords(tokenizedSentence,words):
        sentenceWord = [stem(word) for word in tokenizedSentence]
        bag = np.zeros(len(words),dtype=np.float32)

        for idx , w in enumerate(words):
            if w in sentenceWord:
                bag[idx] = 1

        return bag

    with open("Data\\Tasks.json",'r') as f:
        Tasks = json.load(f)

    allWords = []
    tags = []
    xy = []

    for Task in Tasks['haveInView']:
        tag = Task['tag']
        tags.append(tag)

        for pattern in Task['patterns']:
            w = tokenize(pattern)
            allWords.extend(w)
            xy.append((w,tag))

    ignoreWords = [',','?','/','.','!']
    allWords = [stem(w) for w in allWords if w not in ignoreWords]
    allWords = sorted(set(allWords))
    tags = sorted(set(tags))

    xTrain = []
    yTrain = []

    for (patternSentence,tag) in xy:
        bag = bagOfWords(patternSentence,allWords)
        xTrain.append(bag)

        label = tags.index(tag)
        yTrain.append(label)

    xTrain = np.array(xTrain)
    yTrain = np.array(yTrain)

    numEpochs = 1000
    batchSize = 8
    learningRate = 0.001
    inputSize = len(xTrain[0])
    hiddenSize = 8
    outputSize = len(tags)

    print(">> Training The TasksExecution :- Working ")

    class ChatDataset(Dataset):

        def __init__(self):
            self.n_samples = len(xTrain)
            self.x_data = xTrain
            self.y_data = yTrain

        def __getitem__(self,index):
            return self.x_data[index],self.y_data[index]

        def __len__(self):
            return self.n_samples
        
    dataset = ChatDataset()

    trainLoader = DataLoader(dataset=dataset,
                                batch_size=batchSize,
                                shuffle=True,
                                num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNet(inputSize,hiddenSize,outputSize).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)

    for epoch in range(numEpochs):
        for (words,labels)  in trainLoader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 ==0:
            print(f'Epoch [{epoch+1}/{numEpochs}], Loss: {loss.item():.4f}')

    print(f'Final Loss : {loss.item():.4f}')

    data = {
    "model_state":model.state_dict(),
    "input_size":inputSize,
    "hidden_size":hiddenSize,
    "output_size":outputSize,
    "allWords":allWords,
    "tags":tags
    }

    FILE = "DataBase\\Tasks.pth"
    torch.save(data,FILE)

    print(f"Training Complete, File Saved To {FILE}")
    print("             ")

TrainTasks()

def TasksExecutor(query):

    class NeuralNet(nn.Module):

        def __init__(self,input_size,hidden_size,num_classes):
            super(NeuralNet,self).__init__()
            self.l1 = nn.Linear(input_size,hidden_size)
            self.l2 = nn.Linear(hidden_size,hidden_size)
            self.l3 = nn.Linear(hidden_size,num_classes)
            self.relu = nn.ReLU()

        def forward(self,x):
            out = self.l1(x)
            out = self.relu(out)
            out = self.l2(out)
            out = self.relu(out)
            out = self.l3(out)
            return out

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open('Data\\Tasks.json','r') as json_data:
        intents = json.load(json_data)

    FILE = "DataBase\\Tasks.pth"
    data = torch.load(FILE)

    input_size = data["input_size"]
    hidden_size = data["hidden_size"]
    output_size = data["output_size"]
    allWords = data["allWords"]
    tags = data["tags"]
    model_state = data["model_state"]

    model = NeuralNet(input_size,hidden_size,output_size).to(device)
    model.load_state_dict(model_state)
    model.eval()

    stemmer = PorterStemmer()

    def tokenize(sentence):
        return nltk.word_tokenize(sentence)

    def stem(word):
        return stemmer.stem(word.lower())

    def bagOfWords(tokenizedSentence,words):
        sentenceWord = [stem(word) for word in tokenizedSentence]
        bag = np.zeros(len(words),dtype=np.float32)

        for idx , w in enumerate(words):
            if w in sentenceWord:
                bag[idx] = 1

        return bag

    sentence = str(query)

    sentence = tokenize(sentence)
    X = bagOfWords(sentence,allWords)
    X = X.reshape(1,X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)

    _ , predicted = torch.max(output,dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output,dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:

        for intent in intents['haveInView']:

            if tag == intent["tag"]:

                reply = random.choice(intent["responses"])
                
                return reply

def TaskInitializer(Query):
    task = str(Query).lower()
    taskNew = str(Query).lower()
    returnData = TasksExecutor(task)

    try:
        if "open" in returnData:
            value = OpenExe(taskNew)
            return value
        elif "whatsapp" in returnData:
            wapString = str(taskNew).replace("send ","")
            wapString = str(wapString).replace("whatsapp ","")
            wapString = str(wapString).replace("message ","")
            wapString = str(wapString).replace("to ","")
            return SendMessage(wapString)
        elif "Nasa" in returnData:
            return [Nasa(), True]
    except:
        pass    