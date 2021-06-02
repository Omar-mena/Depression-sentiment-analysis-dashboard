import tweepy
API_KEY = '8xvUEF0lAb858eEnGWu0eYH5j'
API_SECRET_KEY = 'sKb1mc2aLkdb2QxFpEXU6VHWiUfOCOUKy9BMJSzkF2abMpjQP3'
ACCESS_TOEKN = '1317910496142569475-CLZoQdkMmOMpH6Xzv3B3SDIxOii8bz'
ACCESS_TOKEN_SECRET = 'xFscaufPjGdMG7Tq555Kp6HwoFDCaZHXJnNWcwKbpDWmf'
auth  = tweepy.OAuthHandler(API_KEY, \
                            API_SECRET_KEY)
auth.set_access_token(ACCESS_TOEKN,  \
                      ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)

TRACK_WORDS = ['depression', 'games']
TABLE_NAME = "Mental_Health"
TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at DATETIME, text VARCHAR(255), \
            depression VARCHAR(255), user_created_at VARCHAR(255), user_location VARCHAR(255), \
            user_description VARCHAR(255), user_followers_count INT, longitude DOUBLE, latitude DOUBLE, \
            retweet_count INT, favorite_count INT"

import pandas as pd
import numpy as np

TrainingData = pd.read_csv(r'C:\Users\eagle\Desktop\training tweets\CombinedTweetsTraining.csv')
TestingData = pd.read_csv(r'C:\Users\eagle\Desktop\training tweets\test1000.csv')

import nltk
import re, string
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from tqdm import tqdm
from nltk.corpus import stopwords
stopwordEn = stopwords.words('english')
from nltk.corpus import wordnet
nltk.download('wordnet')
import re, string, unicodedata
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


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

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = tokenization(words)
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.preprocessing import FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

# Level: lexicon, model: tf-idf
text_clf = Pipeline([

    ('vect', CountVectorizer(analyzer=normalize and lemmatize_sentence_And_POStag)),
    # ('tfidf', TfidfTransformer(use_idf=True)),

    ('clf', SGDClassifier())

])

text_clf.fit(TrainingData.Text, TrainingData.Sentiment)
predicted = text_clf.predict(TestingData.text)
print(predicted)

from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(TestingData.Sentiment, predicted)*100)
target_names = ['negative', 'positive']
print(metrics.classification_report(TestingData.Sentiment, predicted, target_names=target_names))
# confusion class
pd.DataFrame(metrics.confusion_matrix(TestingData.Sentiment, predicted),
             columns=target_names,index=target_names)

import re
import tweepy
import mysql.connector
import pandas as pd
from textblob import TextBlob
from io import StringIO


# Streaming With Tweepy
# http://docs.tweepy.org/en/v3.4.0/streaming_how_to.html#streaming-with-tweepy


# Override tweepy.StreamListener to add logic to on_status
class MyStreamListener(tweepy.StreamListener):
    '''
    Tweets are known as “status updates”. So the Status class in tweepy has properties describing the tweet.
    https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object.html
    '''

    def on_status(self, status):
        '''
        Extract info from tweets
        '''

        if status.retweeted:
            # Avoid retweeted info, and only original tweets will be received
            return True
        # Extract attributes from each tweet
        id_str = status.id_str
        created_at = status.created_at
        text = deEmojify(status.text)  # Pre-processing the text
        # TO DO next
        # sentiment = TextBlob(text).sentiment
        # polarity = sentiment.polarity
        # subjectivity = sentiment.subjectivity
        # pd.DataFrame(text, columns=["text"])
        print(text)
        text = text.replace("\n", " ")

        depression = text_clf.predict(pd.DataFrame(StringIO(text), columns=["text"]).text)
        depression = "".join(str(x) for x in depression)
        # depression = depression.tostring()
        # print(depression)

        # print(type(depression))

        user_created_at = status.user.created_at
        user_location = deEmojify(status.user.location)
        user_description = deEmojify(status.user.description)
        user_followers_count = status.user.followers_count
        longitude = None
        latitude = None
        if status.coordinates:
            longitude = status.coordinates['coordinates'][0]
            latitude = status.coordinates['coordinates'][1]

        retweet_count = status.retweet_count
        favorite_count = status.favorite_count

        print(status.text)
        print("Long: {}, Lati: {}".format(longitude, latitude))

        # Store all data in MySQL
        if mydb.is_connected():
            mycursor = mydb.cursor()
            sql = "INSERT INTO {} (id_str, created_at, text, depression, user_created_at, user_location, user_description, user_followers_count, longitude, latitude, retweet_count, favorite_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)".format(
                TABLE_NAME)
            val = (id_str, created_at, text, depression, user_created_at, user_location, \
                   user_description, user_followers_count, longitude, latitude, retweet_count, favorite_count)
            mycursor.execute(sql, val)
            mydb.commit()
            mycursor.close()

    def on_error(self, status_code):
        '''
        Since Twitter API has rate limits, stop srcraping data as it exceed to the thresold.
        '''
        if status_code == 420:
            # return False to disconnect the stream
            return False


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


mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="moria",
    database="TwitterDB",
    charset = 'utf8'
)
if mydb.is_connected():
    '''
    Check if this table exits. If not, then create a new one.
    '''
    mycursor = mydb.cursor()
    mycursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_name = '{0}'
        """.format(TABLE_NAME))
    if mycursor.fetchone()[0] != 1:
        mycursor.execute("CREATE TABLE {} ({})".format(TABLE_NAME, TABLE_ATTRIBUTES))
        mydb.commit()
    mycursor.close()

myStreamListener = MyStreamListener()
print("——————————————")
myStream = tweepy.Stream(auth = api.auth, listener = myStreamListener)
print("============================")

myStream.filter(languages=["en"], track = TRACK_WORDS)

# Close the MySQL connection as it finished
# However, this won't be reached as the stream listener won't stop automatically
# Press STOP button to finish the process.
mydb.close()

# Load data from MySQL to perform exploratory data analysis
import mysql.connector
import pandas as pd
import time
import itertools
import math

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
# %matplotlib inline
import plotly.express as px
import datetime
from IPython.display import clear_output

import plotly.offline as py
import plotly.graph_objs as go
from plotly.subplots import make_subplots

py.init_notebook_mode()

import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Filter constants for states in US
# STATES = ['Alabama', 'AL', 'Alaska', 'AK', 'American Samoa', 'AS', 'Arizona', 'AZ', 'Arkansas', 'AR', 'California', 'CA', 'Colorado', 'CO', 'Connecticut', 'CT', 'Delaware', 'DE', 'District of Columbia', 'DC', 'Federated States of Micronesia', 'FM', 'Florida', 'FL', 'Georgia', 'GA', 'Guam', 'GU', 'Hawaii', 'HI', 'Idaho', 'ID', 'Illinois', 'IL', 'Indiana', 'IN', 'Iowa', 'IA', 'Kansas', 'KS', 'Kentucky', 'KY', 'Louisiana', 'LA', 'Maine', 'ME', 'Marshall Islands', 'MH', 'Maryland', 'MD', 'Massachusetts', 'MA', 'Michigan', 'MI', 'Minnesota', 'MN', 'Mississippi', 'MS', 'Missouri', 'MO', 'Montana', 'MT', 'Nebraska', 'NE', 'Nevada', 'NV', 'New Hampshire', 'NH', 'New Jersey', 'NJ', 'New Mexico', 'NM', 'New York', 'NY', 'North Carolina', 'NC', 'North Dakota', 'ND', 'Northern Mariana Islands', 'MP', 'Ohio', 'OH', 'Oklahoma', 'OK', 'Oregon', 'OR', 'Palau', 'PW', 'Pennsylvania', 'PA', 'Puerto Rico', 'PR', 'Rhode Island', 'RI', 'South Carolina', 'SC', 'South Dakota', 'SD', 'Tennessee', 'TN', 'Texas', 'TX', 'Utah', 'UT', 'Vermont', 'VT', 'Virgin Islands', 'VI', 'Virginia', 'VA', 'Washington', 'WA', 'West Virginia', 'WV', 'Wisconsin', 'WI', 'Wyoming', 'WY']
STATES = ['England', 'Wales', 'Scotland', 'Ireland']
STATE_DICT = dict(itertools.zip_longest(*[iter(STATES)] * 2, fillvalue=""))
INV_STATE_DICT = dict((v, k) for k, v in STATE_DICT.items())

'''
This complex plot shows the latest Twitter data within 20 mins and will automatically update.
'''
while True:
    clear_output()
    db_connection = mysql.connector.connect(
        host="localhost",
        user="root",
        passwd="moria",
        database="TwitterDB",
        charset='utf8'
    )
    # Load data from MySQL
    timenow = (datetime.datetime.utcnow() - datetime.timedelta(hours=0, minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
    query = "SELECT id_str, text, created_at, depression, user_location FROM {} WHERE created_at >= '{}' " \
        .format(TABLE_NAME, timenow)
    df = pd.read_sql(query, con=db_connection)
    # UTC for date time at default
    df['created_at'] = pd.to_datetime(df['created_at'])

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[1, 0.4],
        row_heights=[0.6, 0.4],
        specs=[[{"type": "scatter", "rowspan": 2}, {"type": "choropleth"}],
               [None, {"type": "bar"}]]
    )

    '''
    Plot the Line Chart
    '''
    # to do polarity to positive or negative
    # Clean and transform data to enable time series
    result = df.groupby([pd.Grouper(key='created_at', freq='2s'), 'depression']).count().unstack(
        fill_value=0).stack().reset_index()
    result = result.rename(
        columns={"id_str": "Num of '{}' mentions".format(TRACK_WORDS[0]), "created_at": "Time in UTC"})
    time_series = result["Time in UTC"][result['depression'] == 'positive'].reset_index(drop=True)
    fig.add_trace(go.Scatter(
        x=time_series,
        y=result[result['depression'] == 'negative'].reset_index(drop=True),
        name="Negative",
        opacity=0.8), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=time_series,
        y=result[result['depression'] == 'positive'].reset_index(drop=True),
        name="Positive",
        opacity=0.8), row=1, col=1)

    '''
    Plot the Bar Chart
    '''
    content = ' '.join(df["text"])
    content = re.sub(r"http\S+", "", content)
    content = content.replace('RT ', ' ').replace('&amp;', 'and')
    content = re.sub('[^A-Za-z0-9]+', ' ', content)
    content = content.lower()

    tokenized_word = word_tokenize(content)
    stop_words = set(stopwords.words("english"))
    filtered_sent = []
    for w in tokenized_word:
        if w not in stop_words:
            filtered_sent.append(w)
    fdist = FreqDist(filtered_sent)
    fd = pd.DataFrame(fdist.most_common(10), columns=["Word", "Frequency"]).drop([0]).reindex()

    # Plot Bar chart
    fig.add_trace(go.Bar(x=fd["Word"], y=fd["Frequency"], name="Freq Dist"), row=2, col=2)
    # 59, 89, 152
    fig.update_traces(marker_color='rgb(59, 89, 152)', marker_line_color='rgb(8,48,107)', \
                      marker_line_width=0.5, opacity=0.7, row=2, col=2)

    '''
    Plot the Geo-Distribution
    '''
    is_in_US = []
    geo = df[['user_location']]
    df = df.fillna(" ")
    for x in df['user_location']:
        check = False
        for s in STATES:
            if s in x:
                is_in_US.append(STATE_DICT[s] if s in STATE_DICT else s)
                check = True
                break
        if not check:
            is_in_US.append(None)

    geo_dist = pd.DataFrame(is_in_US, columns=['State']).dropna().reset_index()
    geo_dist = geo_dist.groupby('State').count().rename(columns={"index": "Number"}) \
        .sort_values(by=['Number'], ascending=False).reset_index()
    geo_dist["Log Num"] = geo_dist["Number"].apply(lambda x: math.log(x, 2))

    geo_dist['Full State Name'] = geo_dist['State'].apply(lambda x: INV_STATE_DICT[x])
    geo_dist['text'] = geo_dist['Full State Name'] + '<br>' + 'Num: ' + geo_dist['Number'].astype(str)
    fig.add_trace(go.Choropleth(
        locations=geo_dist['State'],  # Spatial coordinates
        z=geo_dist['Log Num'].astype(float),  # Data to be color-coded
        locationmode='country names',  # set of locations match entries in `locations`
        colorscale="Blues",
        text=geo_dist['text'],  # hover text
        showscale=False,
        geo='geo'
    ),
        row=1, col=2)

    fig.update_layout(
        title_text="Real-time tracking '{}' mentions on Twitter {} UTC".format(TRACK_WORDS[0],
                                                                               datetime.datetime.utcnow().strftime(
                                                                                   '%m-%d %H:%M')),
        geo=dict(
            scope='europe',
        ),
        template="plotly_dark",
        margin=dict(r=20, t=50, b=50, l=20),
        annotations=[
            go.layout.Annotation(
                text="Source: Twitter",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0,
                y=0)
        ],
        showlegend=False,
        xaxis_rangeslider_visible=True
    )

    fig.show()

    time.sleep(60)
