import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
file_name = 'embedding_model.h5'
loaded_model_tree = pickle.load(open(file_name, "rb"))

import json
import random
import os

from response_funtion import model_predict, load_predict



cur_dir = os.getcwd() + '\CHATBOT_FOOD'
print(cur_dir)
intents = json.loads(open(cur_dir+'\intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

class Chat:
    egative_responses = ("nothing", "don't", "stop", "sorry")

    exit_commands = ("quit", "pause", "exit", "goodbye", "bye", "later")

    def __init__(self):
        self.name = ''
        self.Buoi = 'Sáng'
        self.Camxuc = 'Vui'
        self.Do_tuoi = 10
        self.So_nguoi = 5
        self.Thoi_tiet = 'Mát'
        self.Phong_cach_am_thuc = 'Việt Nam'
        self.Loai_hinh_quan_an = 'Nhà Hàng Sang Trọng'
        self.Che_do_an = 'ăn thoải mái'
        self.Dac_biet = 'Có chỗ để xe'
        self.Do_pho_bien = 'hot trend'
        self.Mon_an = 'miến gà'
        

    def welcome(self):
        self.name = input('Bot: Trước tiên hãy cho tôi biết tên của bạn để dễ giao tiếp hơn\n\nYou: ')
        self.Buoi = input(f'Bot: Chào {self.name}, hãy cung cấp cho chúng tôi bạn đang ở buổi nào trong ngày (Sáng, Trưa, Chiều, Tối, Đêm)\n\nYou: ')
        self.Do_tuoi = input(f'Bot: Đang là buổi {self.Buoi}. Vậy bạn bao nhiêu tuổi\n\nYou: ')
        self.Camxuc = input(f'Bot: Vậy cảm xúc của bạn đang như thế nào\n\nYou: ')
        self.So_nguoi = input(f'Bot: Có tất cả bao nhiêu người đi ăn vậy\n\nYou: ')
        self.Thoi_tiet = input(f'Bot: Thời tiết ở chỗ bạn như thế nào\n\nYou: ')
        self.Phong_cach_am_thuc = input(f'Bot: Bạn muốn ăn phong cách ẩm thực nào\n\nYou: ')
        self.Loai_hinh_quan_an = input(f'Bot: Vậy còn loại hình quán ăn mà bạn muốn ăn\n\nYou: ')
        self.Che_do_an = input(f'Bot: Chế độ ăn của bạn như thế nào\n\nYou: ')
        self.Dac_biet = input(f'Bot: Bạn có yêu cầu đặc biệt gì không?, ví dụ như có chỗ để xe\n\nYou: ')
        self.Do_pho_bien = input(f'Bot: Bạn muốn độ phổ biến của món ăn như thế nào\n\nYou: ')
        self.Mon_an = load_predict(model_predict(Buoi=self.Buoi, Cam_xuc=self.Camxuc, Do_tuoi=self.Do_tuoi, So_nguoi=self.So_nguoi,
                                                Thoi_tiet=self.Thoi_tiet,Phong_cach_am_thuc=self.Phong_cach_am_thuc,
                                                Loai_hinh_quan_an=self.Loai_hinh_quan_an, Che_do_an=self.Che_do_an,
                                                Dac_biet=self.Dac_biet, Do_pho_bien=self.Do_pho_bien))
        reply = input('Bot: Cảm ơn bạn đã cung cấp thông tin. Giờ đây hãy hỏi tôi những gì bạn thắc mắc \n\nYou: ')
        self.handle_conversation(reply)

    
    def make_exit(self, reply):
        for exit_command in self.exit_commands:
            if exit_command in reply:
                print("Ok, have a great day!")
                return True
        return False

    def handle_conversation(self, reply):
        while not self.make_exit(reply):
            reply = self.chatbot_response(reply)

    def clean_up_sentence(self, sentence):
        # tokenize the pattern 
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word 
        sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

    def bow(self, sentence, words, show_details=True):
        sentence_words = self.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    # assign 1 if current word is in the vocabulary position
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)
        return(np.array(bag))

    def predict_class(self, sentence, model):
        p = self.bow(sentence, words,show_details=False)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list

    def getResponse(self, ints, intents_json):
        tag = ints[0]['intent']
        if tag == 'kết quả':
            return f'Bot: Tôi đã tính toán và đưa ra kết quả, bạn hãy thử {self.Mon_an} nhé\n'
        if tag == 'cụ thể':
            tag = self.Mon_an
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag'] == tag):
                result = random.choice(i['responses'])
                break
        return 'Bot: ' + result + '\n\n'

    def chatbot_response(self, msg):
        ints = self.predict_class(msg, model)
        res = self.getResponse(ints, intents)
        return input(res + '\n\nYou: ')

chat = Chat()

os.system('cls')
chat.welcome()

# #Creating GUI with tkinter
# import tkinter
# from tkinter import *


# def send():
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     EntryBox.delete("0.0",END)

#     if msg != '':
#         ChatLog.config(state=NORMAL)
#         ChatLog.insert(END, "You: " + msg + '\n\n')
#         ChatLog.config(foreground="#000000", font=("Verdana", 12 ))
    
#         res = chat.chatbot_response(msg)
#         ChatLog.insert(END, "Bot: " + res + '\n\n')
            
#         ChatLog.config(state=DISABLED)
#         ChatLog.yview(END)
 

# base = Tk()
# base.title("FoodChatBot")
# base.geometry("400x500")
# base.resizable(width=FALSE, height=FALSE)

# #Create Chat window
# ChatLog = Text(base, bd=0, bg="#dfe6e9", height="8", width="50", font="Tahoma",fg ='#000000',)

# ChatLog.config(state=DISABLED)

# #Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set

# #Create Button to send message
# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
#                     bd=0, bg="#34495e", activebackground="#2c3e50",fg='#ffffff',
#                     command= send )

# #Create the box to enter message
# EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
# #EntryBox.bind("<Return>", send)
# EntryBox.bind("<Return>", lambda event: send())


# #Place all components on the screen
# scrollbar.place(x=376,y=6, height=386)
# ChatLog.place(x=6,y=6, height=386, width=370)
# EntryBox.place(x=128, y=401, height=90, width=265)
# SendButton.place(x=6, y=401, height=90)

# base.mainloop()
