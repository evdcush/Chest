#!/usr/bin/env python

import os
import sys
import code
import shutil
import subprocess
from datetime import datetime
import fire
import pyperclip
import utils
from utils import NOTIMPLEMENTED
from utils import READMES, AWESOME_LISTS, HOARD_ARCHIVE # paths


#=============================================================================#
#                                                                             #
#         888    888          888                                             #
#         888    888          888                                             #
#         888    888          888                                             #
#         8888888888  .d88b.  888 88888b.   .d88b.  888d888 .d8888b           #
#         888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"   88K               #
#         888    888 88888888 888 888  888 88888888 888     "Y8888b.          #
#         888    888 Y8b.     888 888 d881 Y8b.     888          X88          #
#         888    888  "Y8888  888 88888P"   "Y8888  888      88888P'          #
#                                 888                                         #
#                                 888                                         #
#                                 888                                         #
#                                                                             #
#=============================================================================#

def rename_extracted(repo_name, rname=None):
    rname = rname if rname else repo_name
    shutil.move(repo_name + '-master', rname)

def rename_zip(rname, zpath="master.zip"):
    # assumes zip is typical "master.zip"
    shutil.move(zpath, rname + '.zip')

def unzip(zip_path=None):
    import zipfile
    zip_path = zip_path if zip_path else "master.zip"
    with zipfile.ZipFile(zip_path) as zfile:
        #==== extract path
        ex_path = None
        zp = zip_path.split('/')
        if len(zp) > 1:
            ex_path = '/'.join(zp[:-1])

        zfile.extractall(ex_path)

@NOTIMPLEMENTED
def rename_zip_and_contents(repo_name, rname, zpath='master.zip'):
    """ rename the zip file AND the zip file dir
    Instead of trying to be slick, just going to
    unzip zipfile
    rename unzipped dir
    zip dir
    del zipfile
    """
    pass


wget_sh = lambda u, fpath: subprocess.run(f"wget {u} -O {fpath}", shell=True)

#-----------------------------------------------------------------------------#
#                             Naming and Pathing                              #
#-----------------------------------------------------------------------------#

# Potential readme file names:
readme_names = ['README.md', 'README', 'README.rst', 'README.txt',
                'readme.md', 'readme', 'readme.txt']

def get_date():
    date  = datetime.now().date()
    year  = str(date.year)
    month = f"{date.month:02}" # MM
    day   = f"{date.day:02}"   # DD
    return '-'.join([year, month, day])

def get_fname_from_repo(url, get_ext=True):
    """ Formats a filename based on the project repo name
    ASSUMES
    -------
    All url inputs are raw links (https://raw.github.com ....)

    Sample
    ------
    input: https://raw.githubusercontent.com/jslee02/awesome-robotics-libraries/master/README.md
    output: awesome-robotics-libraries.md
    """
    split_url = url.split('/master/') # [raw/user/repo, file.txt]
    raw_name = split_url[-1]
    repo_name = split_url[0].split('/')[-1]
    if get_ext:
        rsplit = raw_name.split(".")
        ext = '' if len(rsplit) == 1 else '.' + rsplit[-1]
    else:
        ext = ''
    fname = repo_name + ext
    return fname.lower()

def make_entry(fname, url):
    date = get_date()
    entry = dict(fname=fname, url=url, date=date)
    return entry


#-----------------------------------------------------------------------------#
#                                  URL stuff                                  #
#-----------------------------------------------------------------------------#
# Domain vars
gh_domain     = "https://github.com"
raw_gh_domain = "https://raw.githubusercontent.com"
raw_len= len(raw_gh_domain + '/')

# funcs
wget_sh = lambda u, fpath: subprocess.run(f"wget {u} -O {fpath}", shell=True)

def check_url_exist(u):
    return subprocess.run(f'wget -q --spider {u}', shell=True).returncode == 0

def check_for_file_link(base_url, candidates):
    u = base_url.replace('github', 'raw.githubusercontent', 1) + '/master/'
    for c in candidates:
        url_candidate = u + c
        if check_url_exist(url_candidate):
            print(f"Found URL link at:\n\t{url_candidate}")
            return url_candidate
    raise FileNotFoundError("No URL found!")

def check_for_readme_links(url):
    readme_link = check_for_file_link(url, readme_names)
    return readme_link

def get_link_from_clipboard():
    url = pyperclip.paste()
    assert url[:18] == gh_domain
    return url


def format_url(url=None):
    """ Formats a url pointing to a file on github to it's raw link

    Params
    ------
    url : str
        url pointing to a file available on github
        eg: 'https://github.com/evdcush/Chest/blob/master/Scripts/get_gh.py'

        if url is None, it is assumed that the url has been copied to the
        clipboard

    Returns
    -------
    u : str
        url to the raw version of the file
        eg: 'https://raw.githubusercontent.com/evdcush/Chest/master/Scripts/get_gh.py'
    """
    if url is None: # get url from clipboard
        url = get_link_from_clipboard()
    # get raw link
    domain = raw_gh_domain
    target = url.split('.com')[-1].replace('blob/', '')
    u = domain + target
    return u


#=============================================================================#
#                                                                             #
#                            888    888                                       #
#                            888    888                                       #
#                            888    888                                       #
#           .d88b.   .d88b.  888888 888888  .d88b.  888d888 .d8888b           #
#          d88P"88b d8P  Y8b 888    888    d8P  Y8b 888P"   88K               #
#          888  888 88888888 888    888    88888888 888     "Y8888b.          #
#          Y88b 888 Y8b.     Y88b.  Y88b.  Y8b.     888          X88          #
#           "Y88888  "Y8888   "Y888  "Y888  "Y8888  888      88888P'          #
#               888                                                           #
#          Y8b d88P                                                           #
#           "Y88P"                                                            #
#                                                                             #
#=============================================================================#
# There is a good amount of redundancy between the getters. Acceptable here,
# for now, as it allows for easier exposure as an interface through fire



"""
USAGE:
 python get_gh.py src https://github.com/google/python-fire/blob/master/docs/guide.md
 python get_gh.py src
 python get_gh.py src --fname='userguide.md'

 # download NiaPy project to cur dir and extract it
 # of course, you need not input url if you copied it!
 python get_gh.py project https://github.com/NiaOrg/NiaPy --dest=None --ex=True
"""

def src(url=None, fname=None, dest=None, archive=True):
    """ get source files, generally code

    >>> url = "https://github.com/seatgeek/fuzzywuzzy/blob/master/fuzzywuzzy/fuzz.py"
    >>> src(url)
    Saving to: ‘seatgeek_fuzzywuzzy_fuzz.py’

    Assumes
    -------
    url is pointing to file url
    """
    # formatting
    target = format_url(url)
    if fname is None:
        usplit = target[raw_len:].split('/')
        user, repo_name = usplit[:2]
        filename = usplit[-1]
        fname = f"{user}_{repo_name}_{filename}"
    fpath = fname if dest is None else f"{dest}/{fname}"

    # get file & archive
    wget_sh(target, fpath)
    if archive:
        entry = make_entry(fpath, target)
        #code.interact(local=dict(globals(), **locals()))
        utils.add_to_archive('src', entry)


def readme(url=None, fname=None, dest=READMES, archive=True, key='readmes'):
    """ get readme from a url
    NB: url may not point to the readme file.
        Since readmes often adhere to a naming format,
        you can crawl for the raw link
    """
    # formatting
    if url is None:
        url = get_link_from_clipboard()
    if "blob" not in url:
        # find link
        target = check_for_readme_links(url)
    else:
        #https://github.com/jslee02/awesome-robotics-libraries/blob/master/README.md
        target = format_url(url)
        # interpret filename
    if fname is None:
        fname = get_fname_from_repo(target)

    fpath = fname if dest is None else f"{dest}/{fname}"

    # get file & archive
    wget_sh(target, fpath)
    if archive:
        entry = make_entry(fpath, target)
        utils.add_to_archive(key, entry)


def alist(url=None, fname=None, dest=AWESOME_LISTS, archive=True, key='awesome_lists'):
    """ get 'awesome' list, or just a file that is a guide/collection

    NB: url may not be pointing to actual file!
        given awesome lists are often in readme files, and
        they are their own projects, the base project url is sufficient to
        crawl for a valid link to the readmefile
    """
    readme(url, fname, dest, archive, key)

"""
# TODO: These are more complicated functions that likely need to use
        github api

def org(url):
    pass

def repo(url):
    pass

def user(url):
    pass

def gist(url):
    pass

"""

def project(url=None, dest=HOARD_ARCHIVE, fname=None, archive=True, ex=False):
    """
    url assumed to be top-level url, NOT archive link
    """
    if url is None:
        url = get_link_from_clipboard()
    user_name, repo_name = url.split('/')[-2:]
    target = url + "/archive/master.zip"

    # get zip
    if fname is None:
        fname = f"{user_name}_{repo_name}"
    zname = fname + ".zip"
    zpath = zname if dest is None else f"{dest}/{zname}"
    wget_sh(target, zpath)

    # extract
    if ex:
        #code.interact(local=dict(globals(), **locals()))
        unzip(zpath)
        # -master is appended, but the rename extract func assumes
        expath = repo_name if dest is None else f"{dest}/{repo_name}"
        rpath  = zpath[:-4] # same name as fname minus '.zip'
        rename_extracted(expath, rpath)

    # archive
    if archive:
        entry = make_entry(fname, url)
        utils.add_to_archive('r', entry)

if __name__ == '__main__':
    fire.Fire(dict(src=src, readme=readme, alist=alist, project=project))


