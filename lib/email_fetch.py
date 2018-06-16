#------------------------------------------------------------------------------
# @author Joshua Rasmussen <xlr8runner@gmail.com>
# @description Function used to get emails after recieving authentication
#   from google servers
#------------------------------------------------------------------------------

import sys
import base64
import json
import bs4
import glob

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

    def get_msg(self, email_id):
        """
        Get the body of a single email based off of the id passe in
        """
        msg = dict()
        msg['id'] = email_id

        body_response = self.resource.get(userId='me', id=email_id).execute()
        msg['body'] = ''

        if 'data' in body_response['payload']['body']:
            msg['body'] = get_text_from_html(_decode(body_response['payload']['body']['data']))
        else:
            for part in body_response['payload']['parts']:
                if 'data' in part['body']:
                    msg['body'] += get_text_from_html(_decode(part['body']['data']))

        for header in body_response['payload']['headers']:
            if header['name'] in ['From', 'Date', 'Subject']:
                msg[header['name']] = header['value']

        return msg

def _decode(data):
    return base64.urlsafe_b64decode(data.encode('ASCII'))

def get_text_from_html(html):
    """
    Takes in a string of html and returns the
    plain text of the document.
    """
    soup = bs4.BeautifulSoup(html, 'html.parser')
    if soup.body:
        return soup.body.get_text()
    else:
        return soup.get_text()

def fetch_and_store_emails():
    """
    Uses the google service to retrieve a collection of Emails
    and immediately stores them into a file
    """
    service = get_gmail_service()
    gmsg = GMessage(service)

    messages = gmsg.get_list()

    num_emails =  len(messages)
    count = 0
    for m in messages:
        sys.stdout.write('\rFetching  ' + str(count) + ' of ' + str(num_emails))
        sys.stdout.flush()
        msg_body = gmsg.get_msg(m['id'])
        _write_to_file(msg_body, m['id'])
        sys.stdout.write('\rCompleted ' + str(count) + ' of ' + str(num_emails))
        sys.stdout.flush()
        count += 1

def _write_to_file(data, id):
    file_name = 'emails/' + str(id) + '.json'
    fp = open(file_name, 'w')
    fp.write(json.dumps(data, sort_keys=True, indent=4, separators=(',',':')))
    fp.close()


def load_emails():
    """
    Load all the stored emails into memory
    """
    emails = []
    email_names = glob.glob('emails/*')
    for email_name in email_names:
        emails.append(_read_email(email_name))

    return emails

def _read_email(file_name):
    fp = open(file_name, 'r')
    return json.loads(fp.read())
