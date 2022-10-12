# Part 3: Mining text data.
# Import packages
import pandas as pd
import csv
import sklearn as sk
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
import re
import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with the string 'wholesale_customers.csv'.
def read_csv_3(data_file):
	"""
	:param data_file: string, the file path
	:return: dataframe, pandas dataframe
	"""
	df = pd.read_csv(data_file, encoding='latin-1')
	return df


# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	"""
	:param df: dataframe, pandas dataframe
	:return: list, the possible sentiments that a tweet might have.
	"""
	# Return a list of the unique value of sentiment columns
	return df['Sentiment'].unique().tolist()


# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	"""
	:param df: dataframe, pandas dataframe
	:return: string, the second most popular sentiment among the tweets
	"""
	# Return the second popular sentiment
	return df['Sentiment'].value_counts().index[1]


# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	"""
	:param df: dataframe, original data frame
	:return: string, the date with the greatest number of extremely positive tweets
	"""
	# Get unique date
	unique_date = df['TweetAt'].unique()

	# Empty dictionary to store sentiment count by date
	date_most_pop_dict = {}

	# Loop over by date
	for date in unique_date:
		tempdf = df[df['TweetAt'] == date]

		# Empty dictionary to store sentiment count
		sentiment_count_dict = {}
		for i in tempdf['Sentiment'].value_counts().index:
			sentiment_count_dict[i] = tempdf['Sentiment'].value_counts()[i]

		# Append date and sentiment count dictionary to date_most_pop_dict
		date_most_pop_dict[date] = sentiment_count_dict

	# Loop over date_most_pop_dict to find out the greatest number of extremely positive tweets
	temp_max = 0
	for k, v in date_most_pop_dict.items():
		if temp_max < int(v['Extremely Positive']):
			temp_max = max(int(d['Extremely Positive']) for d in date_most_pop_dict.values())

	# Return the date that has the greatest number of extremely positive tweets
	for k, v in date_most_pop_dict.items():
		for k1, v1 in v.items():
			if v1 == temp_max:
				return k


# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	"""
	Modify the dataframe to lower case.
	:param df: dataframe, original dataframe
	"""
	# Change all the strings to lower case of the original tweets
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()


# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	"""
	Replace each characters which is not alphabetic or whitespace with a whitespace
	:param df: dataframe, original dataframe
	"""
	# Reference: https://www.codegrepper.com/code-examples/python/remove+non-alphabetic+pandas+python
	df['OriginalTweet'] = df['OriginalTweet'].str.replace('[^a-z]', ' ', regex=True)


# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	"""
	Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
	:param df: dataframe. After remove the character which are not alphabetic or whitespaces.
	"""
	# Reference: https://stackoverflow.com/questions/43071415/remove-multiple-blanks-in-dataframe
	df['OriginalTweet'] = df['OriginalTweet'].replace('/\s\s+/g', ' ', regex=True)


# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	"""
	Tokenize every tweet by converting it into a list of words (strings).
	:param df: dataframe. After remove the character which are not alphabetic or whitespaces.
	"""
	# Convert the strings to a list of words by tokenizing.
	df.OriginalTweet = df.OriginalTweet.str.split()


# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	"""
	:param tdf: dataframe. Dataframe with column 'OriginalTweet' tokenized.
	:return: integer. The number of words in all tweets including repetitions
	"""
	# Variable to store the total number of words with repetitions.
	total_number_of_words_with_repetitions = 0

	# Loop over tdf
	for i in range(len(tdf)):

		# Add the whole length of the tweet to the variable total number of words with repetitions.
		total_number_of_words_with_repetitions += len(tdf['OriginalTweet'][i])

	# Return the amount of word with repetition.
	return total_number_of_words_with_repetitions


# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	"""
	:param tdf: dataframe. Dataframe with column 'OriginalTweet' tokenized.
	:return: integer. The number of words in all tweets without repetitions
	"""
	# Empty list to store all the words
	tweet_with_word_repetitions = []

	# Loop over the dataframe and add all the words to the empty list
	for i in range(len(tdf)):
		original_tweet = tdf['OriginalTweet'][i]
		for word in original_tweet:
			tweet_with_word_repetitions.append(word)

	# Extract distinct word.
	tweet_without_word_repetitions = set(tweet_with_word_repetitions)

	# Return the length of distinct word sets.
	return len(tweet_without_word_repetitions)


# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
	"""
	:param tdf: dataframe. Dataframe with column 'OriginalTweet' tokenized.
	:param k: integer. The k distinct words.
	:return: list. The k distinct words that are most frequent in the tweets.
	"""
	# Empty list to store all the words
	tweet_with_word_repetitions = []

	# Loop over the dataframe and add all the words to the empty list
	for i in range(len(tdf)):
		original_tweet = tdf['OriginalTweet'][i]
		for word in original_tweet:
			tweet_with_word_repetitions.append(word)

	# Count the word frequency and change the pandas series to a list.
	word_frequent_list = pd.Series(tweet_with_word_repetitions).value_counts().sort_values(ascending=False).index.tolist()

	# Return the k distinct words that are most frequent in the tweets
	return word_frequent_list[0:k]


# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	"""
	:param tdf: dataframe. Dataframe with column 'OriginalTweet' tokenized.
	Remove stop words and words with <=2 characters from each tweet.
	"""
	# Reference: https://stackoverflow.com/questions/21546739/load-data-from-txt-with-pandas
	# Read the stop words from a URL to a dataframe and to a list
	url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	stop_word_df = pd.read_fwf(url, delimiter="\t")
	stop_word_df = stop_word_df.rename({'a': 'StopWord'}, axis=1)
	stop_word_df = stop_word_df.append({'StopWord': 'a'}, ignore_index=True)
	stop_word_list = stop_word_df['StopWord'].tolist()

	# Loop over the dataframe
	for i in range(len(tdf)):
		
		# Empty list to store the tweet without stopwords and words with <=2 characters
		without_stop_word_tweet = []
		original_tweet = tdf['OriginalTweet'][i]

		# Loop over the tweet
		for j in range(len(original_tweet)):

			# Remove stop words and words with <=2 characters from each tweet
			if original_tweet[j] not in stop_word_list and len(original_tweet[j]) > 2:
				without_stop_word_tweet.append(original_tweet[j])

		# Replace the Original tweet.
		tdf['OriginalTweet'][i] = without_stop_word_tweet


# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	"""
	:param tdf: dataframe. Original
	Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
	"""
	# Reference: https://towardsdatascience.com/a-beginners-guide-to-stemming-in-natural-language-processing-34ddee4acd37
	# Loop over the dataframe
	for i in range(len(tdf)):

		# Get the original tweet
		original_tweet = tdf['OriginalTweet'][i]

		# Save the stemmer
		stemmer = PorterStemmer()

		# Replace the word with the stem word.
		stems = [stemmer.stem(w) for w in original_tweet]

		# Replace the original tweet with stems
		tdf['OriginalTweet'][i] = stems


# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	"""
	:param df: dataframe. Original dataframe
	:return: array. Predicted sentiments
	"""
	# Reference: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
	# Reference: https://towardsdatascience.com/sentiment-analysis-introduction-to-naive-bayes-algorithm-96831d77ac91
	token = RegexpTokenizer(r'[a-zA-Z0-9]+')

	# Create count vectorizer, the highest accuracy is with the parameters as follow:
	# strip_accents = 'unicode', stop_words = 'english', ngram_range = (1, 3)
	vectorizer = CountVectorizer(strip_accents = 'unicode', stop_words = 'english', ngram_range = (1, 3))
	X = vectorizer.fit_transform(df['OriginalTweet'])
	MNB = MultinomialNB()

	# Fit the whole dataframe to MNB
	MNB.fit(X=X, y=df['Sentiment'])

	# Return the prediction.
	return MNB.predict(X)


# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
	"""
	:param y_pred: array. Predicted sentiment
	:param y_true: array. True sentiment
	:return: float. Sentiment prediction accuracy of training sets.
	"""
	# Variable to store the correct number of prediction
	correct_amount = 0

	# Loop over the y_pred array.
	for i in range(len(y_pred)):

		# If there is same prediction, correct amount +1
		if y_pred[i] == y_true[i]:
			correct_amount += 1

	# Return the correct percentage. Rounded in the 3rd decimal digit.
	return round((correct_amount / len(y_true)), 3)


# Testing
# data_file = 'coronavirus_tweets.csv'
# df = read_csv_3(data_file)
# # print(df)
# y_p = mnb_predict(df)
# # df['Sentiment'].array
# # print(y_p.index())
# acc = mnb_accuracy(y_p, df['Sentiment'].array)
# print(acc)
# print(p.ndim)
#
# data_file = 'coronavirus_tweets.csv'
# df1 = read_csv_3(data_file)
# lower_case(df1)
# remove_non_alphabetic_chars(df1)
# remove_multiple_consecutive_whitespaces(df1)
# tokenize(df1)
# temp = count_words_with_repetitions(df1)
# temp2 = count_words_without_repetitions(df1)
# print(temp)
# print(temp2)
# print(df1['OriginalTweet'])


