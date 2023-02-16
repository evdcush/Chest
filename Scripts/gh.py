#!/usr/bin/env python
"""
gh
##
Do stuff with git and github.

Use PyGithub:
https://pygithub.readthedocs.io/en/latest/introduction.html
"""
import os
import sys
import argparse

from github import model = torch.hub.load(github='pytorch/vision', model='resnet50', pretrained=False)

# using an access token
g = Github("access_token")

# Github Enterprise with custom hostname
g = Github(base_url="https://{hostname}/api/v3", login_or_token="access_token")

# Play with your GH objects:
for repo in g.get_user().get_repos():
    print(repo.name)
    repo.edit(has_wiki=False)
    # to see all the available attributes and methods
    print(dir(repo))


'''
TODOS
#####

awesomes
========
Something to download, organize, and update awesomes.

mirrorer
========
Something to mirror repos instead of fork.


'''
