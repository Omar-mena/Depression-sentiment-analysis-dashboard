# This is Main function.
# Extracting streaming data from Twitter, pre-processing, and loading into MySQL
import credentials  # Import api/access_token keys from credentials.py
import settings  # Import related setting constants from settings.py
import os
import psycopg2
import tweepy
import pandas as pd
from joblib import load
from io import StringIO
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stopwordEn = stopwords.words('english')
nltk.download('wordnet')
import nltk
from nltk.tag import pos_tag
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
stopwordEn = stopwords.words('english')
nltk.download('wordnet')
import re, unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# Override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    '''

    '''

    def on_status(self, status):
        '''
        Extract info from tweets
        '''
        clf = load('depression_model1.joblib') # loading the sentiment analysis model that as SGDclassifier we created in the jupyter notebook
        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        id_str = status.id_str
        created_at = status.created_at
        text = deEmojify(status.text)  # Pre-processing the ext to remove emojis
        text = text.replace("\n", " ")
        print("=====================")
        print(text)
        depression = clf.predict(pd.DataFrame(StringIO(text), columns=["text"]).text) # predicting the sentiment (depression) from the text which is the tweet
        depression = "".join(str(x) for x in depression)
        user_created_at = status.user.created_at
        user_location = deEmojify(status.user.location) # Pre-processing
        user_description = deEmojify(status.user.description) # Pre-processing
        user_followers_count = status.user.followers_count
        longitude = None
        latitude = None
        if status.coordinates:
            longitude = status.coordinates['coordinates'][0] #attaching coordinates to longitude
            latitude = status.coordinates['coordinates'][1] #attaching coordinates to latitude

        retweet_count = status.retweet_count
        favorite_count = status.favorite_count

        # Store all data collected from the streaming in the Heroku PostgreSQL database
        cur = conn.cursor()
        sql = "INSERT INTO {} (id_str, created_at, text, depression, user_created_at, user_location, user_description, user_followers_count, longitude, latitude, retweet_count, favorite_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)".format(settings.TABLE_NAME)
        val = (id_str, created_at, text, depression, user_created_at, user_location, \
                user_description, user_followers_count, longitude, latitude, retweet_count, favorite_count)
        cur.execute(sql, val)
        conn.commit()

        delete_query = '''
         DELETE FROM {0}
         WHERE id_str IN (
             SELECT id_str
             FROM {0}
             ORDER BY created_at asc
             LIMIT 200) AND (SELECT COUNT(*) FROM Mental_Health) > 9600;
         '''.format(settings.TABLE_NAME)

        cur.execute(delete_query)
        conn.commit()
        cur.close()

    def on_error(self, status_code):
        '''
        Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.
        '''
        if status_code == 420:
            # return False to disconnect the stream
            return False


def tokenization(word):
  word = nltk.word_tokenize(word)
  return word


def lemmatize_sentence_And_POStag(tokens):
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    #stem_tokens = porter.stem(tokens)
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        elif tag.startswith('ADJ'):
            pos = 'j'
        else:
            pos = 'a'
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word1 = re.sub(r'[^\w\s]', '', word)
        new_word = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b|@\w+|#', '', new_word1, flags=re.MULTILINE)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words


def normalize(words):
    words = tokenization(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = remove_stopwords(words)
    return words

def clean_tweet(self, tweet):
    '''
    Use sumple regex statemnents to clean tweet text by removing links and special characters
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                |(\w+:\/\/\S+)", " ", tweet).split())


def deEmojify(text):
    '''
    Strip all non-ASCII characters to remove emoji characters
    '''
    if text:
        return text.encode('ascii', 'ignore').decode('ascii')
    else:
        return None


DATABASE_URL = os.environ['DATABASE_URL']

conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cur = conn.cursor()
'''
uncomment this if the table doesnt exist
'''

#cur.execute("""
#        SELECT COUNT(*)
#        FROM information_schema.tables
#        WHERE table_name = '{0}'
#        """.format(settings.TABLE_NAME))
#if cur.fetchone()[0] == 0:
#    cur.execute("CREATE TABLE {} ({});".format(settings.TABLE_NAME, settings.TABLE_ATTRIBUTES))
#    conn.commit()
#cur.close()

# connecting to the twitter api, and start the process of streaming
auth = tweepy.OAuthHandler(credentials.API_KEY, credentials.API_SECRET_KEY)
auth.set_access_token(credentials.ACCESS_TOEKN, credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
print("========================")
print(auth)
print(api)
myStreamListener = MyStreamListener()
myStream = tweepy.Stream(auth=api.auth, listener=myStreamListener)
myStream.filter(languages=["en"], track=settings.TRACK_WORDS)
conn.close()

