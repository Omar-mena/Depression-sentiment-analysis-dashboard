# Depression-sentiment-analysis-dashboard
web app url:https://sentimentanalysis78875.herokuapp.com/
## Introduction
This project is aiming to apply Real-time sentiment analysis on Twitter to detect depression during the Covid-19 pandemic. Natural language preprocessing techniques are used on the data, feature extraction uses POS tagging to help with the lemmatisation, and the trained sentiment analysis model uses SVM that is optimised with SGD to obtain good prediction results. The accuracy for seen and unseen datasets is 94% and 75%, respectively, indicating that the model can classify tweets based on identifying depression. The model became the core of a dashboard web app; the data visualisation element in the app provided the proper graphs for Real-time sentiment analysis, and that includes a time series to show the tweets creation time and compares the classification over 1 minute to 40 mins. To conclude, the app is working successfully 24/7, but it will not keep the data on the database, so to see mental health changes over time in the Covid-19 period, a daily observation from the user is necessary. 
