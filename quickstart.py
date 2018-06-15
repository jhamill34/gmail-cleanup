from __future__ import print_function
import nltk
import re
import base64
import bs4
import sys
import time
import glob
import json

import matplotlib.pyplot as plt
import numpy as np

# Pandas
import pandas as pd

# Required libraries to fetch emails
from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

# Libraries for NLP
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.decomposition import PCA

stopwords = nltk.corpus.stopwords.words('english')
stemmer = SnowballStemmer('english')

########################
# Gmail Authentication #
########################

def get_gmail_service():
    """
    Authenticate useing client_secret
    returns a service object
    """

    # Setup the Gmail API
    SCOPES = 'https://www.googleapis.com/auth/gmail.readonly'
    store = file.Storage('credentials.json')
    creds = store.get()
    if not creds or creds.invalid:
        flow = client.flow_from_clientsecrets('client_secret.json', SCOPES)
        creds = tools.run_flow(flow, store)
    service = build('gmail', 'v1', http=creds.authorize(Http()))

    return service

##########################
# Fetch Emails and Store #
##########################

class GMessage:
    """
    Class that handles the requests to get gmail data
    """
    def __init__(self, service):
        self.resource = service.users().messages()

    def get_list(self):
        """
        Class that retrieves the list of emails
        """
        req = self.resource.list(userId='me')
        messages = req.execute()

        return messages.get('messages')

    def get_body(self, email_id):
        """
        Get the body of a single email based off of the id passe in
        """
        body_response = self.resource.get(userId='me', id=email_id).execute()
        body = ''

        if 'data' in body_response['payload']['body']:
            body = get_text_from_html(decode(body_response['payload']['body']['data']))
        else:
            for part in body_response['payload']['parts']:
                if 'data' in part['body']:
                    body += get_text_from_html(decode(part['body']['data']))

        return body

    def get_msg(self, email_id):
        """
        Get the body of a single email based off of the id passe in
        """
        msg = dict()
        msg['id'] = email_id

        body_response = self.resource.get(userId='me', id=email_id).execute()
        msg['body'] = ''

        if 'data' in body_response['payload']['body']:
            msg['body'] = get_text_from_html(decode(body_response['payload']['body']['data']))
        else:
            for part in body_response['payload']['parts']:
                if 'data' in part['body']:
                    msg['body'] += get_text_from_html(decode(part['body']['data']))

        for header in body_response['payload']['headers']:
            if header['name'] in ['From', 'Date', 'Subject']:
                msg[header['name']] = header['value']

        return msg

def decode(data):
    return base64.urlsafe_b64decode(data.encode('ASCII'))

def get_text_from_html(html):
    soup = bs4.BeautifulSoup(html, 'html.parser')
    if soup.body:
        return soup.body.get_text()
    else:
        return soup.get_text()

def write_to_file(data, id):
    file_name = 'emails/' + str(id) + '.json'
    fp = open(file_name, 'w')
    fp.write(json.dumps(data, sort_keys=True, indent=4, separators=(',',':')))
    fp.close()

def fetch_and_store_emails():
    service = get_gmail_service()
    gmsg = GMessage(service)

    messages = gmsg.get_list()

    num_emails =  len(messages)
    count = 0
    for m in messages:
        sys.stdout.write('\rFetching  ' + str(count) + ' of ' + str(num_emails))
        sys.stdout.flush()
        msg_body = gmsg.get_msg(m['id'])
        write_to_file(msg_body, m['id'])
        sys.stdout.write('\rCompleted ' + str(count) + ' of ' + str(num_emails))
        sys.stdout.flush()
        count += 1

def read_email(file_name):
    fp = open(file_name, 'r')
    return json.loads(fp.read())

def load_emails():
    emails = []
    email_names = glob.glob('emails/*')
    for email_name in email_names:
        emails.append(read_email(email_name))

    return emails

#######################
# Document Clustering #
#######################

# Taken from http://brandonrose.org/clustering
def tokenize_and_stem(text):
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
    #define vectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(max_df=5, min_df=2, stop_words='english')

    tfidf_matrix = tfidf_vectorizer.fit_transform(docs) #fit the vectorizer to synopses
    terms = tfidf_vectorizer.get_feature_names()

    return (tfidf_matrix, terms)

def document_similarities(tfidf_matrix):
    dist = 1 - cosine_similarity(tfidf_matrix)
    return dist

def document_cluster(matrix, n, load_from_file=True):
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
def top_tfidf_feats(row, features, top_n=20):
    topn_ids = np.argsort(row)[::-1][:top_n]
    top_feats = [(features[i], row[i]) for i in topn_ids]
    df = pd.DataFrame(top_feats, columns=['features', 'score'])
    return df

def top_feats_in_doc(X, features, row_id, top_n=25):
    row = np.squeeze(X[row_id].toarray())
    return top_tfidf_feats(row, features, top_n)

def top_mean_feats(X, features, grp_ids=None, min_tfidf=0.1, top_n=25):
    if grp_ids:
        D = X[grp_ids].toarray()
    else:
        D = X.toarray()

    D[D < min_tfidf] = 0
    tfidf_means = np.mean(D, axis=0)
    return top_tfidf_feats(tfidf_means, features, top_n)

#################
# Main Routines #
#################

def main():
    # Load information from files
    emails = load_emails()

    # Create term frequencies
    email_df = pd.DataFrame(emails)
    matrix, terms = generate_term_freqencies(email_df['body'])

    print(top_mean_feats(matrix, terms))

    matrix_dense = matrix.todense()
    coords = PCA(n_components=2).fit_transform(matrix_dense)

    # plt.scatter(coords[:, 0], coords[:, 1], c='m')
    # plt.show()

if __name__ == "__main__":
    main()
