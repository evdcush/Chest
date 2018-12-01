# -*- coding: utf-8 -*-
"""
The MIT License (MIT): Copyright (c) 2016 maguowei
ACKNOWLEDGEMENT: Source script from github.com/maguowei
"""
import code
import os
import yaml
from io import BytesIO
from argparse import ArgumentParser
from collections import OrderedDict

from github3 import GitHub
from github3.exceptions import NotFoundError


# Pathing and Token
# -----------------
HOME = os.environ["HOME"]
token_path = f'{HOME}/.Dots/gh_tokens.yml'

# Grab token
with open(token_path) as tpath:
    _tokens = yaml.load(tpath) # <----<< MUST GIT-CRYPT UNLOCK FIRST
    TOKEN = _tokens['public']['Scrape']['token']

# Argparse for user
# -----------------
P = ArgumentParser()
adg = P.add_argument
adg('-u', '--username', default='evdcush')
adg('-t', '--token',    default=TOKEN)
adg('-s', '--sort', action='store_true', help='sort stars by language')

# README contents
# ---------------
desc = ("# :stars: Star Map\n"
        "A list of {user}'s' starred repos.\n\n"
        "## Contents\n")

CC0_button = ('http://mirrors.creativecommons.org'
             '/presskit/buttons/88x31/svg/cc-zero.svg')

license_ = ("## Licence\n"
            f"[![CC0]({CC0_button})]"
            "(https://creativecommons.org/publicdomain/zero/1.0/)\n"
            "This work is released in the public domain; "
            "no copyright or related rights are reserved over this work\n\n"
            "## Acknowledgement\n"
            "*Original generator script from "
            "[maguowei's starred](https://github.com/maguowei/starred)*")

# HTML formatting
# ---------------
html_escape_table = {">": "&gt;",
                     "<": "&lt;",}
def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c, c) for c in text)

# Stars
# -----
def starred(user, token, sort_results=False):
    """ Creates a Github-style awesome list from user stars

    Params
    ------
    user : str
        github username from whose stars the list will be made
    token : str
        github Oauth token for API; no-privilege public token
    sort_results : bool
        whether to sort results alphabetically (under language)

    Example
    -------
    # Make sorted awesome list of cookiemonster's stars
    python starred.py -u cookiemonster -s > README.md

    """
    # Get stars
    # ---------
    gh = GitHub(token=token)
    stars = gh.starred_by(user)

    # Start building .md output
    print(desc.format(user=user))
    repo_dict = {}

    # Process user stars
    # ------------------
    for s in stars:
        language = s.language or 'Others'
        description = ''
        if s.description:
            description = html_escape(s.description).replace('\n', '')
        if language not in repo_dict:
            repo_dict[language] = []
        repo_dict[language].append([s.name, s.html_url, description.strip()])

    # Sort stars
    if sort_results:
        repo_dict = OrderedDict(sorted(repo_dict.items(), key=lambda l: l[0]))

    # Output contents
    # ---------------
    # Language headings
    for language in repo_dict.keys():
        k = language
        v = '-'.join(language.lower().split())
        data = f'  - [{k}](#{v})'
        print(data)
        print('')

    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
    for language, repos in repo_dict.items():
        print(f"## {language.replace('#', '# #')}\n")
        for repo_name, repo_url, repo_desc in repos:
            data = f"- [{repo_name}]({repo_url}) - {repo_desc}"
            print(data)
        print('')

    print(license_)

if __name__ == '__main__':
    args = vars(P.parse_args())
    user  = args['username']
    token = args['token']
    sort_results = args['sort']
    starred(user, token, sort_results)

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
