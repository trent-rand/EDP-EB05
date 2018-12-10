import gzip
import gensim
import logging
import os

import numpy as np
import pandas as pd

import networkx as nx

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file which is in gzip format"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if (i % 10000 == 0):
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "reviews_data_minimus_reduced.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    # save only the word vectors
    #model.wv.save("C:/Users/Trent Rand/Documents/EDP-EB05/EDP-EB05-master/vectors/default")


    #vec = get_mean_vector(model, doc.words)
    print(model.wv);

    word_vectors = model.wv.vocab

    X = []
    for word in model.wv.vocab:
        X.append(model.wv[word])

    X = np.array(X)

    print("Computed X: ", X.shape)


    #my_words = [word for word in my_words if word in model] # filter out words not in model

# The number of connections we want: either as a factor of the number of words or a set number
    num_top_conns = 150

#######

# Make a list of all word-to-word distances [each as a tuple of (word1,word2,dist)]
    dists=[]

## Method 1 to find distances: use gensim to get the similarity between each word pair
    for i1,word1 in enumerate(word_vectors):
	       for i2,word2 in enumerate(word_vectors):
		             if i1>=i2: continue
		             cosine_similarity = model.similarity(word1,word2)
		             cosine_distance = 1 - cosine_similarity
		             dist = (word1, word2, cosine_distance)
		             dists.append(dist)

## Or, Method 2 to find distances: use scipy (faster)
#from scipy.spatial.distance import pdist,squareform
#Matrix = np.array([model[word] for word in my_words])
#dist = squareform(pdist(Matrix,'cosine'))
#for i1,word1 in enumerate(my_words):
#$	for i2,word2 in enumerate(my_words):
#		if i1>=i2: continue
#		cosine_distance = Matrix[i1, i2]
#		dist = (word1, word2, cosine_distance)
#		dists.append(dist)

######

# Sort the list by ascending distance
    dists.sort(key=lambda _tuple: _tuple[-1])

# Get the top connections
    top_conns = dists[:num_top_conns]

    g = nx.Graph()
    #g.relabel_nodes(model.wv[word])
    for word1,word2,dist in top_conns:
	       weight = 1 - dist # cosine similarity makes more sense for edge weight
	       g.add_edge(word1, word2, weight=float(weight))

# Write the network
    nx.write_graphml(g, 'Word2VecGenSimYelpReviews.graphml')

    nx.draw(g, pos=nx.random_layout(g), with_labels=True, nodecolor='b', edge_color='r')
    #nx.relabel_nodes(model.wv.vocab)
    plt.show()
    #G = nx.Graph()  # build graph
    #G.add_nodes_from(model.wv)  # add nodes
    #G.add_edges_from()    #add edges


    #X = model[model.wv.vocab]
    #pca = PCA(n_components=2)
    #result = pca.fit_transform(X)
    # create a scatter plot of the projection
    #pyplot.scatter(result[:, 0], result[:, 1])
    #words = list(model.wv.vocab)
    #for i, word in enumerate(words):
	#       pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    #pyplot.show()

    ######### TEST ##########
    def get_mean_vector(word2vec_model, words):
        # remove out-of-vocabulary words
        words = [word for word in words if word in word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(word2vec_model[words], axis=0)
        else:
            return []
