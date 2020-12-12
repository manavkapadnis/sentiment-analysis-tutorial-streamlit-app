#Necessary imports
import streamlit as st
import pandas as pd
import webbrowser
import pickle
import regex as re
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer,PorterStemmer,SnowballStemmer
from nltk.sentiment.util import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.gaussian_process import GaussianProcessClassifier
from nltk.corpus import words

nltk.download('wordnet')
nltk.download('punkt')


#Headings for Web Application
st.title("Sentiment Analysis Using Count Vectorizer And Tf-idf")
st.subheader("This application will show different parts of sentiment analysis :")
st.markdown("1. Tokenization And Stopwords Removal\n")
st.markdown("2. Stemming And Lemmatization\n")
st.markdown("3. Feature Extraction And Implementing Different Classifiers\n")

st.sidebar.subheader('About the Creator:')
st.sidebar.markdown('Manav Nitin Kapadnis')
st.sidebar.markdown('Sophomore | IIT Kharagpur')

github_url = 'https://github.com/manavkapadnis'
if st.sidebar.button('Github'):
    webbrowser.open_new_tab(github_url)

linkedin_url = 'https://www.linkedin.com/in/manav-nitin-kapadnis-013b94192/'
if st.sidebar.button('LinkedIn'):
    webbrowser.open_new_tab(linkedin_url)

k_url = 'https://www.kaggle.com/darthmanav'
if st.sidebar.button('Kaggle'):
    webbrowser.open_new_tab(k_url)

#Textbox for text user is entering
st.subheader("Enter the text you'd like to analyze:")
text = st.text_input('Please enter text which is more than 10 words') #text is stored in this variable
text_copy=text

st.subheader('1. Tokenization And Stopwords Removal')
st.markdown('''
Tokenization is the process by which big quantity of text is divided into smaller parts called tokens.\n
It is crucial to understand the pattern in the text in order to perform various NLP tasks.These tokens are very useful for finding such patterns.\n
\n
Natural Language toolkit has very important module tokenize which further comprises of sub-modules \n
\n
1.word tokenize\n
2.sentence tokenize\n
''')


def w_token(text):
	words = nltk.word_tokenize(text)
	st.success(words)
	st.write("The number of tokens are", len(words))
	unique_tokens = set(words)
	st.write("The number of unique tokens are",len(unique_tokens))


def s_token(text):
	words = nltk.sent_tokenize(text)
	st.success(words)
	st.write("The number of sentences are",len(words))


def remove_words(text):
	words = nltk.word_tokenize(text)
	stop_words = set(stopwords.words('english'))
	final_tokens = []
	for each in words:
		if each not in stop_words:
			final_tokens.append(each)
	st.write("The number of total tokens after removing stopwords are",len((final_tokens)))

if st.button("Show Word Tokenization"):
	w_token(text_copy)

if st.button("Show Sentence Tokenization"):
	s_token(text_copy)

if st.button("Remove Stopwords"):
	remove_words(text_copy)	


st.subheader('2. Stemming And Lemmatization')
st.markdown('''
**Stemming** is a kind of normalization for words. Normalization is a technique where a set of words in a sentence are converted into a sequence to shorten its lookup. The words which have the same meaning but have some variation according to the context or sentence are normalized.\n
Hence Stemming is a way to find the root word from any variations of respective word\n

There are many stemmers provided by Nltk like **PorterStemmer**, **SnowballStemmer**, **LancasterStemmer**.\n
**Lemmatization** is the algorithmic process of finding the lemma of a word depending on their meaning. Lemmatization usually refers to the morphological analysis of words, which aims to remove inflectional endings. It helps in returning the base or dictionary form of a word, which is known as the lemma.\n
The NLTK Lemmatization method is based on WorldNet's built-in morph function.\n
We will see differences between Porter stemmer, Snowball stemmer, and Lemmatization.\n
''')

def combo_function(text):
	stemmer_ps = PorterStemmer()
	stemmer_ss = SnowballStemmer("english")
	lemmatizer = WordNetLemmatizer()  
	words_1=nltk.word_tokenize(text)
	words_2=nltk.word_tokenize(text)
	words_3=nltk.word_tokenize(text)
	words_4=nltk.word_tokenize(text)
	stemmed_words_ps = [stemmer_ps.stem(word) for word in words_1]
	stemmed_words_ss = [stemmer_ss.stem(word) for word in words_2]
	lemmatized_words = [lemmatizer.lemmatize(word, pos = "v") for word in words_3]
	st.write(pd.DataFrame({
		'tokens': words_4,
		'Porterstemmer': stemmed_words_ps,
		'Snowballstemmer': stemmed_words_ss,
		'lemmatized words': lemmatized_words
		}))



if st.button("Show Stemming and Lemmatization alongwith original text"):
	combo_function(text_copy)

st.subheader('3. Feature Extraction And running the classifiers')
st.markdown('After running the different classifiers, **MLP classifier** performs best for **Count Vectorizer** Feature Extraction and **Gaussian Process Classifier** works best for **Tf-idf Vectorizer**')
st.markdown('The metrics above are for training and validating on Restaurant reviews dataset.\n')

option = st.selectbox('Which feature extraction would you like to use?', ('Count Vectorizer', 'Tf-idf Vectorizer'))

if option == 'Count Vectorizer':
	st.markdown('''For Count-Vectorizer Feature Extraction :\n
		The accuracy score of MLP classifier is: 0.8\n
		The precision score of MLP classifier is: 0.7864\n
		The recall of MLP classifier is: 0.8182\n
		The F1 score of MLP classifier is: 0.8019\n
		The roc auc score of MLP classifier is: 0.8001800180018003\n
		''')
	
	loaded_model_cv = pickle.load(open("review_cv.pkl", "rb"))
	cv= pickle.load(open("cv.pkl", "rb"))
	def new_review(new_review):
		new_review = new_review
		new_review = re.sub('[^a-zA-Z]', ' ', new_review)
		new_review = new_review.lower()
		new_review = new_review.split()
		ps = PorterStemmer()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		new_review = [ps.stem(word) for word in new_review if not word in  set(all_stopwords)]
		new_review = ' '.join(new_review)
		new_corpus = [new_review]
		new_X_test = cv.transform(new_corpus).toarray()
		#print(new_X_test.shape)
		new_y_pred = loaded_model_cv.predict(new_X_test)
		return new_y_pred
	new_review = new_review(text)
	if new_review[0]==1:
		st.success("The entered text is Positive")
	else :
		st.success("The entered text is Negative")

if option == 'Tf-idf Vectorizer':
	st.write('''For Tf-idf Vectorizer Feature Extraction :\n
		The accuracy score of Gaussian Process Classifier is: 0.80\n
		The precision score of Gaussian Process classifier is: 0.7087\n
		The recall of Gaussian Process classifier is: 0.8795\n
		The F1 score of Gaussian Process classifier is: 0.7849\n
		The roc auc score of Gaussian Process classifier is: 0.8115\n
		''')
	
	loaded_model_tv = pickle.load(open("review_tv.pkl", "rb"))
	tv=pickle.load(open("tv.pkl", "rb"))
	def new_review(new_review):
		new_review = new_review
		new_review = re.sub('[^a-zA-Z]', ' ', new_review)
		new_review = new_review.lower()
		new_review = new_review.split()
		ps = PorterStemmer()
		all_stopwords = stopwords.words('english')
		all_stopwords.remove('not')
		new_review = [ps.stem(word) for word in new_review if not word in  set(all_stopwords)]
		new_review = ' '.join(new_review)
		new_corpus = [new_review]
		new_X_test = tv.transform(new_corpus).toarray()
		#print(new_X_test.shape)
		new_y_pred = loaded_model_tv.predict(new_X_test)
		return new_y_pred
	new_review = new_review(text)
	if new_review[0]==1:
		st.success("The entered text is Positive")
	else :
		st.success("The entered text is Negative")






