import gzip
import gensim
import logging
import os

import networkx as nx

from sklearn.decomposition import PCA
from matplotlib import pyplot


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
    data_file = os.path.join(abspath, "reviews_data_further_reduced.txt.gz")

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

    w1 = "dirty"
    print("Most similar to {0}".format(w1), model.wv.most_similar(positive=w1))

    # look up top 6 words similar to 'polite'
    w1 = ["polite"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'france'
    w1 = ["france"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["shocked"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # look up top 6 words similar to 'shocked'
    w1 = ["beautiful"]
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            topn=6))

    # get everything related to stuff on the bed
    w1 = ["bed", 'sheet', 'pillow']
    w2 = ['couch']
    print(
        "Most similar to {0}".format(w1),
        model.wv.most_similar(
            positive=w1,
            negative=w2,
            topn=10))

    # similarity between two different words
    print("Similarity between 'dirty' and 'smelly'",
          model.wv.similarity(w1="dirty", w2="smelly"))

    # similarity between two identical words
    print("Similarity between 'dirty' and 'dirty'",
          model.wv.similarity(w1="dirty", w2="dirty"))

    # similarity between two unrelated words
    print("Similarity between 'dirty' and 'clean'",
          model.wv.similarity(w1="dirty", w2="clean"))

    #vec = get_mean_vector(model, doc.words)
    #print(vec);



    #G = nx.Graph()  # build graph
    #G.add_nodes_from()  # add nodes
    #G.add_edges_from()    #add edges


    X = model[model.wv.vocab]
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    # create a scatter plot of the projection
    pyplot.scatter(result[:, 0], result[:, 1])
    words = list(model.wv.vocab)
    for i, word in enumerate(words):
	       pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
    pyplot.show()

    ######### TEST ##########
    def get_mean_vector(word2vec_model, words):
        # remove out-of-vocabulary words
        words = [word for word in words if word in word2vec_model.vocab]
        if len(words) >= 1:
            return np.mean(word2vec_model[words], axis=0)
        else:
            return []
