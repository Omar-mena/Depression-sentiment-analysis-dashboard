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