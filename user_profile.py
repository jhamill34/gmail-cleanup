from __future__ import print_function

import json

from lib import gmail_auth as ga
from lib import util

def main():
    service = ga.get_gmail_service()

    profile = service.users().getProfile(userId='me').execute()
    print(util.json_pretty(profile))

if __name__ == '__main__':
    main()
