#importing required modules
import pandas as pd
import string
import re
from textblob import TextBlob
import nltk
nltk.download('stopwords')

df1 = pd.read_json(r'tweets1.json', lines=True)
df1 = df1[["created_at", "id", "text"]]
df1.to_csv(r'tweets1.csv')


df1['text'] = df1.text.apply(lambda x: x.lower())
df1['text'] = df1.text.apply(lambda x: x.translate(string.punctuation))
df1['text'] = df1.text.apply(lambda x: x.translate(string.digits))


def tokenization(text):
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split()
    return text

df1['tokenized_tweet'] = df1['text'].apply(lambda x: tokenization(x))
print(df1.tokenized_tweet)
print('#'*100)

stopword = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text = [word for word in text if word not in stopword]
    text = [word for word in text if word != 'rt']
    return text
    
df1['nonstop_tweet'] = df1['tokenized_tweet'].apply(lambda x: remove_stopwords(x))
print(df1.nonstop_tweet)
print('#'*100)

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df1['stemmed_tweet'] = df1['nonstop_tweet'].apply(lambda x: stemming(x))
df1['stemmed_tweet'] = df1['stemmed_tweet'].replace('rt','', regex=True)
df1['clean_tweet'] = df1['stemmed_tweet'].apply(lambda x: ' '.join(x))

print(df1.clean_tweet)
print('#'*100)

df1.to_csv(r'path/cleaned_tweets.csv')

import pandas as pd
import numpy as np
import tweepy # to use Twitter’s API
from textblob import TextBlob # for doing sentimental analysis
import re # regex for cleaning the tweets
import nltk
nltk.download('brown')
nltk.download('punkt')

df2 = pd.read_csv(r'cleaned_tweets.csv')
df2 = df2[['created_at', 'text', 'clean_tweet']]
print(df2.clean_tweet)

def polarity_calc(text):
    try:
        return TextBlob(text).sentiment.polarity
    except:
        return None

def subjectivity_calc(text):
    try:
        return TextBlob(text).sentiment.subjectivity
    except:
        return None

df2['polarity'] = df2['clean_tweet'].apply(polarity_calc)
df2['subjectivity'] = df2['clean_tweet'].apply(subjectivity_calc)


df2.to_csv(r'path/sent.csv')
print(df2.head)
