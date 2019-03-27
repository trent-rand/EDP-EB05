import re
import numpy as np
import pandas as pd
from pprint import pprint
import networkx as nx
from networkx.readwrite import json_graph
from networkx.readwrite import edgelist
from networkx.readwrite import adjlist




# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# Let's go get our friend WikiRelate:
#from folder.wikirelate as wikirelate

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

G = nx.read_edgelist("test1.edgelist")
topicJSONData = json_graph.node_link_data(G)


output = open("topicedgelist.txt", "w")

for line in adjlist.generate_adjlist(G):
    output.write("%s" % line)

    line = line.strip().split(" ")
    ind = 0;
    newline = ""

    for char in line:
        if char == "-":
            char = "1"

        newline = newline + char

        if ind == 0:
            newline = newline + ","+ str(len(line)) +":"
        elif ind == len(line):
            newline = newline + ",1\n"
        else:
            newline = newline + ",1:"

        ind = ind + 1

    #output.write("%s" % newline)


output.close()


#import json
#with open('topicEdgeData.json', 'w') as outfile:
#    json.dump(topicEDGEData, outfile)

#topicJSONGraph = json_graph.node_link_graph(G)
#json.dump(topicJSONGraph, 'topicNodeGraph.json')
