from __future__ import print_function

import os

import pandas as pd
from sklearn.decomposition import PCA

from lib import gmail_auth as ga
from lib import email_fetch as ef
from lib import clustering as cl
from twilio.rest import Client

# Your Account SID from twilio.com/console
account_sid = os.environ['TWILIO_SID']

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
        to=os.environ['TWILIO_TO'],
        from_=os.environ['TWILIO_FROM'],
        body="Downloaded " + str(number_of_files) + " emails from your account!")



if __name__ == "__main__":
    fetch_emails()
