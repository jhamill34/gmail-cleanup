#------------------------------------------------------------------------------
# @author Joshua Rasmussen <xlr8runner@gmail.com>
# @description Functions to vectorize the emails and run clustering
#   on them.
#------------------------------------------------------------------------------

import nltk
import re

# Pandas
import pandas as pd
import numpy as np

# Libraries for NLP
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

#######################
# Document Clustering #
#######################

# Taken from http://brandonrose.org/clustering
def tokenize_and_stem(text):
    """
    Uses nltk's tokenize database to tokenize a body of text by
    sentences and words. Words are also broken down into their stems
    """
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems

def generate_term_freqencies(docs):
    """
    Takes a collection of documents and determines an approprate
    word frequency matrix.
    """
    #define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=5, min_df=2, stop_words='english')

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs) #fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    return (tfidf_matrix, terms)

def _document_similarities(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    return dist

def document_cluster(matrix, n, load_from_file=True):
    """
    Takes a term frequency matrix and runs unsupervised learning
    clustering methods on it to gain insights on the colletion.
    When done it stores into a pickle.
    """
    if load_from_file:
        km = joblib.load('cluster_fit.pkl')
    else:
        km = KMeans(n_clusters=n)
        km.fit(matrix)
        joblib.dump(km, 'cluster_fit.pkl')

    return km

#####################
# Top Term Extrator #
#####################
def _top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df

def top_feats_in_doc(X, features, row_id, top_n=25):
    """
    Looking a single document returns the top_n number of
    features located in the document
    """
    row = np.squeeze(X[row_id].toarray())
    return _top_tfidf_feats(row, features, top_n)

def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    """
    Looks at all the documents in the corpus and returns the top_n
    features in the collection.
    """
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return _top_tfidf_feats(tfidf_means, features, top_n)
