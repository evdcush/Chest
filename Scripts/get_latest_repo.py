"""
Get latest release from a github repo
# URL
#-----------------------
# user/repo-name/releases[/latest]

get_release_level = lambda b: '' if b else '/latest'

# parser for get url
#-----------------------
class Parser:
    P = argparse.ArgumentParser()
    # argparse doesn't like bool, so this is the workaround
    p_bool = {'type': int, 'default': 0, 'choices':[0,1]}

    def __init__(self):
        adg = self.P.add_argument
        adg('--user',        '-u', type=str, required=True)
        adg('--repo',        '-r', type=str, required=True)
        adg('--pre_release', '-p', **self.p_bool) # bools

    def parse_args(self):
        parsed = AttrDict(vars(self.P.parse_args()))
        parsed.pre_release = bool(parsed.pre_release)
        return parsed

    def __call__(self):
        parsed = self.parse_args()
        user, repo = parsed.user, parsed.repo
        release_lvl = get_release_level(parsed.pre_release)
        self.url = url = URL.format(user, repo, release_lvl)
        return url

"""

from json import loads
import argparse
from urllib.request import urlopen, urlretrieve, Request
from pprint import pprint

# Attribute dict (dot access)
#-----------------------
class AttrDict(dict): # just a dict mutated/accessed by attribute instead index
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

###############################################################################
#                                                                             #
#  888888888888        ,ad8888ba,        88888888ba,          ,ad8888ba,      #
#       88            d8"'    `"8b       88      `"8b        d8"'    `"8b     #
#       88           d8'        `8b      88        `8b      d8'        `8b    #
#       88           88          88      88         88      88          88    #
#       88           88          88      88         88      88          88    #
#       88           Y8,        ,8P      88         8P      Y8,        ,8P    #
#       88            Y8a.    .a8P       88      .a8P        Y8a.    .a8P     #
#       88             `"Y8888Y"'        88888888Y"'          `"Y8888Y"'      #
#                                                                             #
"""############################################################################

Features:
---------
- specify download destination
- what file within assets?
- whether to unpack/unzip etc



"""############################################################################
# URL
#-----------------------
# user/repo-name/releases[/latest]
URL_BASE = 'https://api.github.com/repos/{}/{}/releases{}'

# parser for user/repo
#-----------------------
parser = argparse.ArgumentParser()
parser.add_argument('--user',        '-u', type=str, required=True)
parser.add_argument('--repo',        '-r', type=str, required=True)
parser.add_argument('--pre_release', '-p', action="store_true")
parsed = parser.parse_args()

# Get url fields
#-----------------------
user, repo = parsed.user, parsed.repo
pre_release = parsed.pre_release
release_level = '' if pre_release else '/latest'
URL = URL_BASE.format(user, repo, release_level)


# function kwarg
#-------------------
headers = {'Accept': 'application/vnd.github.v3+json'}

# fGET
#-------------------
json = loads(urlopen(Request(URL, headers=headers)).read())
#pprint(json)
asset = json[0]['assets'][0]
pprint('assets: {}'.format(asset))
urlretrieve(asset['browser_download_url'], asset['name'])
