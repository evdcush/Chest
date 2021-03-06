#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The MIT License (MIT): Copyright (c) 2016 maguowei
ACKNOWLEDGEMENT: Source script from github.com/maguowei
"""
import os
#import yaml
from argparse import ArgumentParser
from collections import OrderedDict

from github3 import GitHub
from github3.exceptions import NotFoundError

from utils import public_gh_tokens

PUBLIC_TOKENS = public_gh_tokens()

# Grab token
#with open(f'{os.environ["HOME"]}/.Dots/gh_tokens.yml') as tpath:
#    _tokens = yaml.load(tpath) # <----<< MUST GIT-CRYPT UNLOCK FIRST
#    TOKEN = _tokens['public']['Scrape']['token']
TOKEN = PUBLIC_TOKENS['Scrape']['token']


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
LINE_BREAK = '\n\n---\n\n'

'''
ffi.write("\n")
        # repo stars
        for lng, projs in sorted(data.items(), key=lambda data: data[0]):
            ffi.write(f"# {lng} \n\n")
            ffi.write("---\n\n")
            for proj in projs:
                pushed_at = datetime.strptime(
                    proj['pushed_at'], "%Y-%m-%dT%H:%M:%SZ")

                last_code_push = str((datetime.utcnow() - pushed_at).days)

                ffi.write(f"## {proj.get('name').strip()}\n\n")
                ffi.write(f"```\n{proj.get('description')}\n```\n\n")
                ffi.write(f"  * [Github]({proj['html_url']})\n")
                ffi.write(f"  * Stars: {proj['stargazers_count']}\n")
                #ffi.write(f"  * Open issues: {proj['open_issues_count']}\n")
                ffi.write("  * Last pushed: {} ({} days)\n\n".format(
                    pushed_at.strftime("%Y-%m-%d"), last_code_push))
'''

license_ = (LINE_BREAK
            "## Licence\n"
            f"[![CC0]({CC0_button})]"
            "(https://creativecommons.org/publicdomain/zero/1.0/)\n"
            "This work is released in the public domain; "
            "no copyright or related rights are reserved over this work"
            LINE_BREAK
            "## Acknowledgement\n"
            "*Original generator script from "
            "[maguowei's starred.py](https://github.com/maguowei/starred)*")

# HTML formatting
# ---------------
html_escape_table = {">": "&gt;",
                     "<": "&lt;",}
def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c, c) for c in text)
WFILE = "starmap.md"
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

    for language, repos in repo_dict.items():
        print(f"## {language.replace('#', '# #')}\n")
        for repo_name, repo_url, repo_desc in repos:
            data = f"- [{repo_name}]({repo_url}) - {repo_desc}"
            print(data)
        print('')
    print(license_)

if __name__ == '__main__':
    args  = P.parse_args()
    user  = args.username
    token = args.token
    sort_results = args.sort
    starred(user, token, sort_results)

