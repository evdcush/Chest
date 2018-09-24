"""
Get latest release from a github repo

"""

import json
import argparse
import urllib.request as req
from pprint import PrettyPrinter

pretty_printer = PrettyPrinter()
pprint = pretty_printer.pprint

# Attribute dict (dot access)
#-----------------------
class AttrDict(dict): # just a dict mutated/accessed by attribute instead index
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

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
URL = 'https://api.github.com/repos/{}/{}/releases{}'
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

# parsing argv and getting url
#--------------------
url_parser = Parser()
url = url_parser()
#urllib.request.urlopen(urllib.request.Request

# function vars
#-------------------
jload = json.loads
urlopen = req.urlopen
request = req.Request
retrieve = req.urlretrieve
headers = {'Accept': 'application/vnd.github.v3+json'}

# fGET
#-------------------
_json = jload(urlopen(request(url, headers=headers)).read())
asset = _json[0]['assets'][0]
pprint('assets: {}'.format(asset))
retrieve(asset['browser_download_url'], asset['name'])


'''
_json = json.loads(req.urlopen(req.Request(
    url,
     headers={'Accept': 'application/vnd.github.v3+json'},
)).read())
'''
