from __future__ import print_function

import pandas as pd
from sklearn.decomposition import PCA

from lib import gmail_auth as ga
from lib import email_fetch as ef
from lib import clustering as cl


#################
# Main Routines #
#################

def main():
    # Load information from files
    emails = ef.load_emails()

    # Create term frequencies
    email_df = pd.DataFrame(emails)
    matrix, terms = cl.generate_term_freqencies(email_df['body'])

    print(cl.top_mean_feats(matrix, terms))

    matrix_dense = matrix.todense()
    coords = PCA(n_components=2).fit_transform(matrix_dense)

if __name__ == "__main__":
    main()
