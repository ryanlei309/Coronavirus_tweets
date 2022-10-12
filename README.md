# Coronavirus_tweets
This is an assignment of Data Mining module when I studied at King's College London Data Science.
I uses the Coronavirus Tweets NLP data set from Kaggle https://www.kaggle.com/ datatattle/covid-19-nlp-text-classification to predict the sentiment of Tweets relevant to Covid.
This coursework is written in Python and use the package of sklearn and nltk to do the analysis.

The following are the steps that I used to predict the sentiment of Tweets relevant to Covid.

1. Compute the possible sentiments that a tweet may have, the second most popular sentiment in the tweets, and the date with the greatest number of extremely positive tweets.
After execute the first part, the results are as follows:

![Screenshot 2022-10-12 at 11 15 41](https://user-images.githubusercontent.com/52303625/195316927-38f1e9fc-b04f-40f0-99bb-698b37876abc.png)

2. Because in the tweet messages, there are non-alphabetical characters. In order to have better accuracy, I wrote functions to handle those situations.

3. Tokenize the tweets (convert each into a list of words), count the total number of all words (including reptitions), and the number of all distinct words and the 10 most frequent words in the corpus.
Also, I recount the 10 most frequent words after removing stop words, words with <= 2 characters and reduce each word to its stem. The results are as follows:

![Screenshot 2022-10-12 at 11 48 31](https://user-images.githubusercontent.com/52303625/195323639-0d104aa7-6006-41d0-8a95-663772e43dcc.png)
![Screenshot 2022-10-12 at 11 48 54](https://user-images.githubusercontent.com/52303625/195323702-a52d6a7a-665b-4882-8739-8f7783a887e0.png)

4. Store the coronavirus tweets.py corpus in a numpy array and produce a sparse representation of the term- document matrix with a CountVectorizer.
Then produce a Multinomial Naive Bayes classifier using the provided data set. The training accuracy of the classifier is 98.9%.

![Screenshot 2022-10-12 at 12 07 04](https://user-images.githubusercontent.com/52303625/195327146-f207ac9b-2194-4c9c-b66f-293e7e2e9109.png)
