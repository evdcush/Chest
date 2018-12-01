# -*- coding: utf-8 -*-
"""
The MIT License (MIT): Copyright (c) 2016 maguowei
ACKNOWLEDGEMENT: Source script from github.com/maguowei
"""
import os
import sys
import yaml
from io import BytesIO
from collections import OrderedDict
import click
from github3 import GitHub
from github3.exceptions import NotFoundError
from argparse import ArgumentParser

HOME = os.environ["HOME"]
token_path = f'{HOME}/.Dots/gh_tokens.yml'

with open(token_path) as tpath:
    _tokens = yaml.load(tpath)
    TOKEN = _tokens['public']['Scrape']['token']


P = ArgumentParser()
adg = P.add_argument
adg('-u', '--username', default='evdcush')
adg('-s', '--sort', action='store_true', help='sort stars by language')
adg('-t', '--token', default=TOKEN)

# starred --username "$USER" --sort --token "$TOKEN" > "$USER.stars.md"


desc = '''# :stars: Star Map

A list of my starred repos.


## Contents
'''

license_ = '''
{}
## Acknowledgement
*Original generator script from [starred](https://github.com/maguowei/starred)*
'''

html_escape_table = {
    ">": "&gt;",
    "<": "&lt;",
}

USER = 'evdcush'

def html_escape(text):
    """Produce entities within text."""
    return "".join(html_escape_table.get(c, c) for c in text)


@click.command()
@click.option('--username', envvar='USER', help='GitHub username')
@click.option('--token', envvar='GITHUB_TOKEN', help='GitHub token')
@click.option('--sort',  is_flag=True, help='sort by language')
@click.option('--message', default='update stars', help='commit message')
@click.version_option(version='2.0.3', prog_name='starred')
def starred():
    """GitHub starred

    creating your own Awesome List used GitHub stars!

    example:
        starred --username maguowei --sort > README.md
    """
    gh = GitHub(token=token)
    stars = gh.starred_by(username)
    click.echo(desc)
    repo_dict = {}

    for s in stars:
        language = s.language or 'Others'
        description = html_escape(s.description).replace('\n', '') if s.description else ''
        if language not in repo_dict:
            repo_dict[language] = []
        repo_dict[language].append([s.name, s.html_url, description.strip()])

    if sort:
        repo_dict = OrderedDict(sorted(repo_dict.items(), key=lambda l: l[0]))

    for language in repo_dict.keys():
        data = u'  - [{}](#{})'.format(language, '-'.join(language.lower().split()))
        click.echo(data)
    click.echo('')

    for language in repo_dict:
        click.echo('## {} \n'.format(language.replace('#', '# #')))
        for repo in repo_dict[language]:
            data = u'- [{}]({}) - {}'.format(*repo)
            click.echo(data)
        click.echo('')

    click.echo(license_.format(username=username))

poo = starred()

#code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
