import tweepy
API_KEY = ''
API_SECRET_KEY = ''
ACCESS_TOEKN = ''
ACCESS_TOKEN_SECRET = ''

auth  = tweepy.OAuthHandler(API_KEY, \
                            API_SECRET_KEY)
auth.set_access_token(ACCESS_TOEKN,  \
                      ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)
