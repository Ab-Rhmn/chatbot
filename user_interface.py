from tkinter import *
from turtle import delay

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
import nltk
nltk.download('stopwords')
from nltk import pos_tag
from nltk.corpus import stopwords
import re
import string
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk.tokenize import sent_tokenize, word_tokenize

intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
#this is the items that we have in the supermarket for now
goods={
    "fruit":1,"vegetable":2,"chicken":3,"dairy":4,"orange":5,"apple":6,"banana":7,"cooky":8,"milk":9,"icecream":10,"chocolate":11,"bread":12
}
final_list=""   #this is to store the shopping cart of the user
def getshelfnumbers(text):
	text = re.sub(r'\d+', '', text)  # numbers elimination
	text = text.translate(str.maketrans('', '', string.punctuation))  # punctuations eliminiation
	text = text.lower()  # lowering
	stop_words = set(stopwords.words('english')) #for eliminating stop words
	word_tokens = word_tokenize(text) #tokenize
	filtered_sentence = [w for w in word_tokens if w not in stop_words] #filter out the stop words
	pos_tagged = pos_tag(filtered_sentence) #pos tag from nltk's pos the filtered words

	tag_map = defaultdict(lambda: wn.NOUN)  #mapping the dict for lemmatizing
	tag_map['JJ'] = wn.ADJ
	tag_map['JJR'] = wn.ADJ
	tag_map['JJS'] = wn.ADJ
	tag_map['VB'] = wn.VERB
	tag_map['VBD'] = wn.VERB
	tag_map['VBG'] = wn.VERB
	tag_map['VBN'] = wn.VERB
	tag_map['VBP'] = wn.VERB
	tag_map['VBZ'] = wn.VERB
	tag_map['NN'] = wn.NOUN				#we can only take nouns but others also taken for clarity here
	tag_map['NNS'] = wn.NOUN
	tag_map['NNP'] = wn.NOUN
	tag_map['NNPS'] = wn.NOUN
	tag_map['RB'] = wn.ADV
	tag_map['RBR'] = wn.ADV
	tag_map['RBS'] = wn.ADV
	tag_map['RP'] = wn.ADV
	lemma_function = WordNetLemmatizer()

	nouns = []  #having different bags for each noun, verb adj and advrb in case if the content is very huge searching only in nouns
	verbs = []  #but here all are taken
	adj = []
	advrb = []
	for token, tag in pos_tagged:  #from the tag map mapping the lemmatization
		lemma = lemma_function.lemmatize(token, tag_map[tag])
		if tag.startswith("N"):
			nouns.append(lemma)
		elif tag.startswith("V"):
			verbs.append(lemma)
		elif tag.startswith("J"):
			adj.append(lemma)
		elif tag.startswith("R"):
			advrb.append(lemma)


	products_needed = []
	products_needed.extend(nouns)# some are tagged as adjevtives also   , if needed can append others also
	products_needed.extend(adj)
	products_needed.extend(advrb)
	products_needed.extend(verbs)

	mylist = {k: goods[k] for k in products_needed if k in goods}  #the product and shelf number in the dictionary
	the_list = ""
	for i in mylist:
		the_list += i + " in the shelf -> " + str(mylist[i]) + '\n'
	global final_list     #adding to final list
	final_list+=the_list
	return the_list   #returning the currently asked products



def bow(sentence):
	global message #making the global variable to access the current text
	message=sentence
	sentence_words = nltk.word_tokenize(sentence)#tokenize the sentence
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]#lemmatize using Wordnet
	bag = [0]*len(words) #bag of words approach instead of tf-idf
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i]=1   #assigning 1 if the word is present
	return (np.array(bag))  #returning the array of bag of words

def predict_class(sentence):  #filter out predictions with minimum threshold
	sentence_bag = bow(sentence)
	res = model.predict(np.array([sentence_bag]))[0] #predict from the trained model
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	#sort by probablity
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent':classes[r[0]], 'probablity':str(r[1])})
	return return_list

def getResponse(ints):  #getting a random response from the intent which has the higher predition value
	tag = ints[0]['intent']
	list_of_intents = intents['intents']
	for i in list_of_intents:
		if(i['tag']==tag):
			result=random.choice(i['responses'])
			break

	return result

def chatbot_response(msg):
	ints = predict_class(msg)
	res = getResponse(ints)
	return res


#this is half learning based for normal ineraction and identifying purpose and rest used bagging for identifying shelves
#once the purpose identified if the product is needed the getshlefnumbers method is called and identified the product shelf number
#and when the conversation is over the entire cart is displayed using the final_list
def send():
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg != '':
		ChatHistory.config(state=NORMAL)
		ChatHistory.insert('end', "You: " + msg + "\n\n")

		res = chatbot_response(msg)
		if res=="The product and their shelf numbers are as follows": #if not normal user interaction doing the shelf search
			res=getshelfnumbers(message)
		elif res=="See you!" or res== "Have a nice day" or res=="Bye! Come back again soon.": #if end of conversation final list
			res=final_list+"\n"+res

		ChatHistory.insert('end', "Bot: " + res + "\n\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')


def view_cart():  #view the entire cart at any time
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg == '':
		ChatHistory.config(state=NORMAL)
		ChatHistory.insert('end', "Your Cart is : "+"\n")
		ChatHistory.insert('end', "Bot: \n" + final_list + "\n\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')

#creating the user interface
base = Tk()
base.title("Supermarket assistant")
base.geometry("400x500")
base.resizable(width=False, height=False)

ChatHistory = Text(base, bd=2, bg='white', font='Arial')
ChatHistory.config(state=DISABLED)

SendButton = Button(base, font=('Arial', 12, 'bold'),
	text="Send", fg="#dfdfdf", activebackground="#3e3e3e", bg="#0000FF", command=send)
viewCartButton = Button(base, font=('Arial', 12, 'bold'),
	text="Cart", fg="#dfdfdf", activebackground="#3e3e3e", bg="#0000FF", command=view_cart)
DisplayBox = Button(base, font=('Arial', 12, 'bold'),
	text="Supermarket Assistant", fg="#dfdfdf", activebackground="#3e3e3e", bg="#0000FF", command=view_cart)

TextEntryBox = Text(base, bd=2, bg='white', font='Arial')
DisplayBox.place(x=6, y=6, height=20, width=386)
ChatHistory.place(x=6, y=26, height=386, width=386)#for the conversation place
TextEntryBox.place(x=128, y=400, height=50, width=265)#for the entering text
SendButton.place(x=6, y=400, height=50, width=50)#for sending the text
viewCartButton.place(x=56, y=400, height=50, width=70)#for viewing the current cart

base.mainloop()
