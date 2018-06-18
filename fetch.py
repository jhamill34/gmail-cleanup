from __future__ import print_function

import pandas as pd
from sklearn.decomposition import PCA

from lib import gmail_auth as ga
from lib import email_fetch as ef
from lib import clustering as cl
from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = "ACb7a4001997135d639d2cd5f6fb548e3d"

# Your Auth Token from twilio.com/console
auth_token  = os.environ['TWILIO_AUTH_TOKEN']

client = Client(account_sid, auth_token)

#################
# Main Routines #
#################

def fetch_emails():
    service = ga.get_gmail_service()
    number_of_files = ef.fetch_and_store_emails(service)

    message = client.messages.create(
        to="+14257609918",
        from_="+14252305280",
        body="Downloaded " + str(number_of_files) + " emails from your account!")



if __name__ == "__main__":
    fetch_emails()
