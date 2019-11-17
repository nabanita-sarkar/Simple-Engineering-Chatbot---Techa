#Meet TECHA: your technical friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages


nltk.download('punkt') 
nltk.download('wordnet') 

fin = "Engineering is the use of scientific principles to design and build machines, structures, and other items, including bridges, tunnels, roads, vehicles, and buildings.[1] The discipline of engineering encompasses a broad range of more specialized fields of engineering, each with a more specific emphasis on particular areas of applied mathematics, applied science, and types of application. See glossary of engineering.The term engineering is derived from the Latin ingenium, meaning cleverness and ingeniare, meaning to contrive, devise.The American Engineers Council for Professional Development (ECPD, the predecessor of ABET)[3] has defined engineering as:The creative application of scientific principles to design or develop structures, machines, apparatus, or manufacturing processes, or works utilizing them singly or in combination; or to construct or operate the same with full cognizance of their design; or to forecast their behavior under specific operating conditions; all as respects an intended function, economics of operation and safety to life and property.Engineering has existed since ancient times, when humans devised inventions such as the wedge, lever, wheel and pulley, etc.The term engineering is derived from the word engineer, which itself dates back to 1390 when an engine'er (literally, one who builds or operates a siege engine) referred to a constructor of military engines. In this context, now obsolete, an engine referred to a military machine, i.e., a mechanical contraption used in war (for example, a catapult). Notable examples of the obsolete usage which have survived to the present day are military engineering corps, e.g., the U.S. Army Corps of Engineers.The word engine itself is of even older origin, ultimately deriving from the Latin ingenium, meaning innate quality, especially mental power, hence a clever invention."
#Reading in the corpus
#with open('Chatbot.gdoc','r', encoding='utf8', errors ='ignore') as fin:
raw = fin.lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+"I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response


flag=True
print("TECHA: My name is Techa. I will answer your queries about Engineering. If you want to exit, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("TECHA: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("TECHA: "+greeting(user_response))
            else:
                print("TECHA: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("TECHA: Bye! take care..")    
    
