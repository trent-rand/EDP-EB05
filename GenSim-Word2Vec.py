import gzip
import gensim
import os

import numpy as np
import pandas as pd

import networkx as nx

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def show_file_contents(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    with gzip.open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            if (i % 10000 == 0):
                yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    abspath = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(abspath, "reviews_data_minimus_reduced.txt.gz")

    # read the tokenized reviews into a list
    # each review item becomes a serries of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=150,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    print(model.wv);

    word_vectors = model.wv.vocab

    X = []
    for word in model.wv.vocab:
        X.append(model.wv[word])

    X = np.array(X)

    print("Computed X: ", X.shape)
    num_top_conns = 150


# Make a list of all topic-to-topic distances [each as a tuple of (word1,word2,dist)]
    dists=[]
    vectors=[]

    def get_mean_vector(word2vec_model, words):
        # remove out-of-vocabulary words
        words = [word for word in words if word in word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(word2vec_model[words], axis=0)
        else:
            return []

    for doc in documents :
        tempVec = get_mean_vector(model, doc)
        vectors.append(tempVec)

## Method 1 to find distances: use gensim to get the similarity between each mean vector pair
    for i1,word1 in enumerate(vectors):
	       for i2,word2 in enumerate(vectors):
		             if i1>=i2: continue
		             cosine_similarity = model.similarity(word1,word2)
		             cosine_distance = 1 - cosine_similarity
		             dist = (word1, word2, cosine_distance)
		             dists.append(dist)

# Sort the list by ascending distance
    dists.sort(key=lambda _tuple: _tuple[-1])

# Get the top connections
    top_conns = dists[:num_top_conns]

    g = nx.Graph()
    for word1,word2,dist in top_conns:
	       weight = 1 - dist # cosine similarity makes more sense for edge weight
	       g.add_edge(word1, word2, weight=float(weight))

    # Write the network
    nx.write_graphml(g, 'Word2VecGenSimYelpReviews.graphml')
    nx.draw(g, pos=nx.random_layout(g), with_labels=True, nodecolor='b', edge_color='r')
    plt.show()
