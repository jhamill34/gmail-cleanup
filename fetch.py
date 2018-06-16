from __future__ import print_function

import pandas as pd
from sklearn.decomposition import PCA

from lib import gmail_auth as ga
from lib import email_fetch as ef
from lib import clustering as cl


#################
# Main Routines #
#################

def fetch_emails():
    service = ga.get_gmail_service()
    ef.fetch_and_store_emails(service, 2)

if __name__ == "__main__":
    fetch_emails()
