# Stock-Sentiment-Analysis

Let's understand the project using STAR technique:

# Situation (S):

Stock market movements are highly influenced by news headlines, and investors often struggle to analyze how news sentiment impacts stock prices. Traditional methods rely on manual analysis, which is time-consuming and prone to biases.

To address this, we developed an automated sentiment analysis model that predicts stock market trends based on financial news headlines.

# Task (T):

The goal of this project was to:
* Analyze financial news headlines to determine market sentiment.
* Preprocess and clean news data for better accuracy.
* Extract features using NLP techniques (Bag of Words).
* Train a machine learning model to classify sentiment.
* Improve accuracy by selecting the right ML model.

# Action (A):

1️. Data Collection & Cleaning

* Libraries Used: pandas, numpy
* Dataset: Collected stock market headlines from 2015 and earlier.

2. Preprocessing:

* Converted text to lowercase for uniformity.
* Combined 25 headline columns into a single text feature.

3. Feature Engineering & Text Processing
   
* Libraries Used: sklearn.feature_extraction.text.CountVectorizer
* Implemented Bag of Words (BoW) Model:
* Used CountVectorizer(ngram_range=(2,2)) to extract bigrams from text data.
* Transformed headlines into numerical feature vectors.
* Created a feature matrix where each row represented a stock news entry.

4. Model Building & Training
   
* Library Used: sklearn.ensemble.RandomForestClassifier
* Selected Machine Learning Model:
* Used RandomForestClassifier for sentiment classification.
* Trained the model on preprocessed news headlines.
* Evaluated performance based on accuracy.

# Result (R):

* ✅ Successfully built an NLP-based stock sentiment classifier with:
* ✅ Cleaned and structured text data for better prediction.
* ✅ Implemented bigram feature extraction to capture context in news headlines.
* ✅ Trained a Random Forest model for sentiment classification.
* ✅ Helped investors analyze how news sentiment affects stock prices.
