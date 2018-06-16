#------------------------------------------------------------------------------
# @author Joshua Rasmussen <xlr8runner@gmail.com>
# @description functions used to become authenticated through the
#   google api system.
#------------------------------------------------------------------------------

from apiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

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
