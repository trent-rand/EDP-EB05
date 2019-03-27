## 3. Import Packages
import re
import numpy as np
import pandas as pd
from pprint import pprint
import networkx as nx

import json

####La7S<AB?5_H=ObFmIo{


# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.test.utils import datapath

# Let's go get our friend WikiRelate:
#from folder.wikirelate as wikirelate

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this


# MySQL Import
import pymysql

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)


## 5. Prepare Stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'timestamp', 'com', 'http', 'www'])


## 6. Import Newsgroups Data
# Import Dataset
connection = pymysql.connect(host='35.203.15.52', user='root', password='5TRcaSeTr4L', db='Twitter')


##
#
#df = pd.read_csv('abcnews-reduced.csv', dtype=str, usecols=[1])

Tweets_df = pd.read_sql('SELECT * FROM Tweets LIMIT 1000000', con=connection)

#print(Tweets_df.iloc[:,0]) #Random IDs
#print(Tweets_df.iloc[:,1]) #Tweet Text Content
#print(Tweets_df.iloc[:,2]) #Timestamps
#print(Tweets_df.iloc[:,3]) #User IDs


#print(Users_df.iloc[:,1])



data = Tweets_df.iloc[:,1].tolist()
#pprint(data[:1])


    ## 8.Tokenize document in GenSim modifiable tokens.
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
#print(data_words[:1])


    ## 9. Creating Bigram and Trigram Models
    # Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

    # Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

    ## 10. Remove Stopwords, Make Bigrams and Lemmatize
    # Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(u' '.join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

    # Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

    # Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    # python3 -m spacy download en
nlp = spacy.load('en', disable=['parser', 'ner'])

    # Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized[:1])

    ## 11. Create the Dictionary and Corpus needed for Topic Modeling
    # Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

    # Create Corpus
texts = data_lemmatized

    # Create corpus bag of words
corpus = [id2word.doc2bow(text) for text in texts]

    # View
    #print(corpus[:1])

    # Human readable format of corpus (term-frequency)
[[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

    ## 12. Building Topic Model (LDA)
    # Build LDA model
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=50,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    ## 13. View topics in LDA model

# Save model to disk.
temp_file = datapath("TweetsLdaModel")
lda_model.save(temp_file)

    # Print the Keyword in the 10 topics
pprint(lda_model.get_topics())
doc_lda = lda_model[corpus]

#pprint(doc_lda)


ndarray = lda_model.get_topics()
pprint(ndarray)


nameList = lda_model.get_topic_terms(2)
pprint(nameList)
