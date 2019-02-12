#!/usr/bin/env python

import os
import sys
import code
import subprocess
from datetime import datetime
import fire
import pyperclip
import utils
from utils import READMES, AWESOME_LISTS, PROJECTS, HOARD_ARCHIVE

# dest dirs
readme_dest = READMES
file_dest   = PROJECTS


#=============================================================================#
#                                                                             #
#         888    888          888                                             #
#         888    888          888                                             #
#         888    888          888                                             #
#         8888888888  .d88b.  888 88888b.   .d88b.  888d888 .d8888b           #
#         888    888 d8P  Y8b 888 888 "88b d8P  Y8b 888P"   88K               #
#         888    888 88888888 888 888  888 88888888 888     "Y8888b.          #
#         888    888 Y8b.     888 888 d88P Y8b.     888          X88          #
#         888    888  "Y8888  888 88888P"   "Y8888  888      88888P'          #
#                                 888                                         #
#                                 888                                         #
#                                 888                                         #
#                                                                             #
#=============================================================================#





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
    return fname

def make_entry(fname, url):
    date = get_date()
    entry = dict(fname=fname, url=url, date=date)
    return date


#-----------------------------------------------------------------------------#
#                                  URL stuff                                  #
#-----------------------------------------------------------------------------#
# Domain vars
gh_domain     = "https://github.com"
raw_gh_domain = "https://raw.githubusercontent.com"

wget_sh = lambda u, fpath: subprocess.run(f"wget {u} -O {fpath}", shell=True)

def check_url_exist(u):
    return subprocess.run(f'wget -q --spider {u}', shell=True).returncode == 0

def check_for_file_link(base_url, candidates):
    u = url.replace('github', 'raw.githubusercontent', 1) + '/master/'
    for c in candidates:
        url_candidate = u + c
        if check_url_exist(url_candidate):
            print(f"Found URL link at:\n\t{url_candidate}")
            return url_candidate
    raise FileNotFoundError("No URL found!")

def check_for_readme_links(url):
    try:
        readme_link = check_for_file_link(url, readme_names)
        return readme_link
    except:
        pass

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
"""
Common Assumptions
------------------
* No input url is a raw link (if you have a raw link, no need to use
  this script, since you already went to the trouble and you can just wget

There is a good amount of redundancy between the getters. Acceptable here,
for now, as it allows for easier exposure as an interface through fire
"""

def src(url=None, fname=None, dest=None, archive=True):
    """ get source files, generally code """
    # formatting
    target = format_url(url)
    if fname is None:
        fname = target.split('/')[-1]
    fpath = fname if dest is None else f"{dest}/{fname}"

    # get file & archive
    wget_sh(target, fpath)
    if archive:
        entry = make_entry(fname, target)
        utils.add_to_archive('src', entry)


def alist(url=None, fname=None, dest=AWESOME_LISTS, archive=True):
    """ get 'awesome' list, or just a file that is a guide/collection

    NB: url may not be pointing to actual file!
        given awesome lists are often in readme files, and
        they are their own projects, the base project url is sufficient to
        crawl for a valid link to the readmefile
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
        entry = make_entry(fname, target)
        utils.add_to_archive('awesome_lists', entry)

#https://gist.github.com/akesling/5358964
#https://gist.githubusercontent.com/akesling/5358964/raw/cc5f54a47d0ffa6b15d2fceb36fa63782abf18c8/mnist.py

#============================================================================================================#
#                                                                                                            #
#  .d8888b.  888                                             888       888    888                            #
# d88P  Y88b 888                                             888       888    888                            #
# Y88b.      888                                             888       888    888                            #
#  "Y888b.   888888  .d88b.  88888b.  88888b.   .d88b.   .d88888       8888888888  .d88b.  888d888  .d88b.   #
#     "Y88b. 888    d88""88b 888 "88b 888 "88b d8P  Y8b d88" 888       888    888 d8P  Y8b 888P"   d8P  Y8b  #
#       "888 888    888  888 888  888 888  888 88888888 888  888       888    888 88888888 888     88888888  #
# Y88b  d88P Y88b.  Y88..88P 888 d88P 888 d88P Y8b.     Y88b 888       888    888 Y8b.     888     Y8b.      #
#  "Y8888P"   "Y888  "Y88P"  88888P"  88888P"   "Y8888   "Y88888       888    888  "Y8888  888      "Y8888   #
#                            888      888                                                                    #
#                            888      888                                                                    #
#                            888      888                                                                    #
#                                                                                                            #
#============================================================================================================#
def gist(url=None, fname=None, dest=READMES, archive=True):
    """ get readme file

    NB: url may not be pointing to actual file! Readme files have a
        common naming format, so links can be crawled
    """
    # check for link
    if url is None:
        url = get_link_from_clipboard()

    # check if link pointing at a file
    if "blob" not in url:
        target = check_for_readme_links(url)
    else: # link to repo
        target = format_url(url)

    # name formatting
    if fname is None:
        fname = get_fname_from_repo(target)
    fpath = fname if dest is None else f"{dest}/{fname}"

    # get file & archive
    wget_sh(target, fpath)
    if archive:
        entry = make_entry(fname, target)
        utils.add_to_archive('readmes', entry)

def readme(url=None, fname=None, dest=READMES, archive=True):
    pass

def org(url):
    pass

def repo(url):
    pass

def user(url):
    pass



#https://github.com/jslee02/awesome-robotics-libraries/blob/master/README.md

#https://github.com/cpp-taskflow/cpp-taskflow/archive/master.zip

def project(url, name=None, dest=file_dest):
    """
    """
    target = format_url(url)





chk_exist = lambda u: subprocess.run(f'wget -q --spider {u}', shell=True)

def wget_readme(url, dest=readme_dest):
    chk_exist = lambda u: subprocess.run(f'wget --spider {u}', shell=True)

    # URL formatting
    u = url.replace('github', 'raw.githubusercontent', 1) + '/master/'
    fname = f'{url.split("/")[-1]}'

    # Check for filename format
    correct_rdme_name = ''
    for rdm in rdme:
        if chk_exist(u + rdm).returncode == 0:
            correct_rdme_name = rdm
            break
    if correct_rdme_name == '':
        print('readme does not match standard readme file names or does not exist')
    else:
        # Get extension of correct readme
        ext = correct_rdme_name.split('.')[-1]
        if ext != correct_rdme_name:
            fname += '.' + ext

        # append true readme name to url
        u = u + correct_rdme_name

        # add url to archive, so readme can be upated
        #add_list_to_archive(fname, u)

        #=== append file write name to dest
        dest = f'{dest}/{fname}'

        #=== Get readme
        subprocess.run(f'wget {u} -O {dest}', shell=True)


#if __name__ == '__main__':
#    args = sys.argv[1:]
#    URL = args.pop(0)
#    #code.interact(local=dict(globals(), **locals()))
#    if args:
#        # args is [filename, dest] or [filename]
#        wget_raw(URL, *args)
#    else:
#        wget_readme(URL)



if __name__ == '__main__':
    fire.Fire()

