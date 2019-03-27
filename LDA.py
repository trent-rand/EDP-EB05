## 3. Import Packages
import re
import os
import numpy as np
import pandas as pd
from pprint import pprint

import networkx as nx

import gzip

import MySQLdb

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

os.environ['MALLET_HOME'] = 'C:\\users\\biney\\mallet_unzipped\\mallet-2.0.8'


def file(x):
    return {
        0:'',
        1:'',
        2:'',
        3:'',
        4:'',
        5:'',
        6:'',
        7:'',
        8:'',
        9:'',
    }.get(x, '')



## 5. Fetch Stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Custom method to parse .gzip corpuses.
def read_gzip(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


    ## 6. Import Data
    # Import Dataset from external files.

#conn = MySQLdb.connect(host="CloudIpAddress", user="username", passwd="password", db="Twitter")

conn = MySQLdb.connect(host="35.203.15.52", user=root, passwd="5TRcaSeTr4L", db="Twitter")

#cursor = conn.cursor()

#cursor.execute('SELECT COUNT(MemberID) as count FROM Members WHERE id = 1')
#table_rows = cursor.fetchall()

df = pd.read_sql('SELECT * FROM Tweets WHERE UserId < 10000', con=conn)

conn.close()

#df = pd.DataFrame(table_rows)

#df = pd.read_csv('abcnews-reduced.csv', dtype=str)
    #abspath = os.path.dirname(os.path.abspath(__file__))
    #df = os.path.join(abspath, file_name)

    ## 7. Convert to List
data = df.values.tolist()
pprint(data[:1])


    ## 8.Tokenize document in GenSim modifiable tokens.
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))
print(data_words[:1])


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
        doc = nlp(" ".join(sent))
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
print(corpus[:1])

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

    # Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

    ## 14. Perplexity + Coherence Score

    # Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
coherence_model_lda = gensim.models.coherencemodel.CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v', )
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

pprint(doc_lda)


ndarray = lda_model.get_topics()
pprint(ndarray)



nameList = lda_model.get_topic_terms(2)
pprint(nameList)


#pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
pyLDAvis.save_html(vis, 'TestRunExport.html')

##################################### MALLET ####################################
