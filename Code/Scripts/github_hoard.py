#!/usr/bin/env python3
# coding=utf-8

# Modules
import argparse
import git
import json
import os
import queue
import requests
import threading
import time
import datetime as dt
import code
import sys
from pprint import pprint

#user_agent = "GithubCloner (https://github.com/mazen160/GithubCloner)"
#headers = {'User-Agent': user_agent, 'Accept': '*/*'}
headers = {'Accept': '*/*'}
timeout = 3


def checkResponse(response):
    """
    Validates whether there an error in the response.
    """
    try:
        if "API rate limit exceeded" in response["message"]:
            print('[!] Error: Github API rate limit exceeded')
            return(1)
            #return(0)
    except TypeError:
        pass
    try:
        if (response["message"] == "Not Found"):
            return(2)  # The organization does not exist
    except TypeError:
        pass
    return(0)
    #return(1)


def fromUser(user, username=None, token=None, include_gists=True):
    """
    Retrieves a list of repositories for a Github user.
    Input:-
    user: Github username.
    Optional Input:-
    username: Github username.
    token: Github token or password.
    Output:-
    a list of Github repositories URLs.
    """
    print(f'fromUser({user})')
    URLs = []
    resp = []
    current_page = 1
    while (len(resp) != 0 or current_page == 1):
        API = "https://api.github.com/users/{0}/repos?per_page=40000000&page={1}".format(user, current_page)
        if (username or token) is None:
            resp = requests.get(API, headers=headers, timeout=timeout).text
        else:
            resp = requests.get(API, headers=headers, timeout=timeout, auth=(username, token)).text
        resp = json.loads(resp)
        if checkResponse(resp) != 0:
            return([])
        for i in range(len(resp)):
            URLs.append(resp[i]["git_url"])
        if include_gists is True:
            URLs.extend(UserGists(user, username=username, token=token))
        current_page += 1
    return(URLs)

def fromOrg(_org_name, username=None, token=None):
    """
    Retrieves a list of repositories for a Github organization.
    Input:-
    org_name: Github organization name.
    Optional Input:-
    username: Github username.
    token: Github token or password.
    Output:-
    a list of Github repositories URLs.
    """
    org_name = _org_name.split('/')[-1]
    print(f'fromOrg({org_name})')
    URLs = []
    resp = []
    current_page = 1
    while (len(resp) != 0 or current_page == 1):
        #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
        API = "https://api.github.com/orgs/{0}/repos?per_page=40000000&page={1}".format(org_name, current_page)
        if (username or token) is None:
            resp = requests.get(API, headers=headers, timeout=timeout).text
        else:
            resp = requests.get(API, headers=headers, timeout=timeout, auth=(username, token)).text
        resp = json.loads(resp)
        if checkResponse(resp) != 0:
            return([])
        for i in range(len(resp)):
            URLs.append(resp[i]["git_url"])
        current_page += 1
    return(URLs)

def cloneRepo(URL, cloningpath, username=None, token=None):
    """
    Clones a single GIT repository.
    Input:-
    URL: GIT repository URL.
    cloningPath: the directory that the repository will be cloned at.
    Optional Input:-
    username: Github username.
    token: Github token or password.
    """
    print(f'fromRepo({URL})')
    try:
        try:
            if not os.path.exists(cloningpath):
                os.mkdir(cloningpath)
        except Exception:
            pass
        URL = URL.replace("git://", "https://")
        if (username or token) is not None:
            URL = URL.replace("https://", "https://{}:{}@".format(username, token))
        repopath = URL.split("/")[-2] + "_" + URL.split("/")[-1]
        if repopath.endswith(".git"):
            repopath = repopath[:-4]
        if '@' in repopath:
            repopath = repopath.replace(repopath[:repopath.index("@") + 1], "")
        fullpath = cloningpath + "/" + repopath
        with threading.Lock():
            print(fullpath)

        if os.path.exists(fullpath):
            git.Repo(fullpath).remote().pull()
        else:
            #git.Repo.clone_from(URL, fullpath)
            git.Repo.clone_from(URL, fullpath, depth=1)
    except Exception:
        print("Error: There was an error in cloning [{}]".format(URL))


def cloneBulkRepos(URLs, cloningPath, threads_limit=5, username=None, token=None):
    """
    Clones a bulk of GIT repositories.
    Input:-
    URLs: A list of GIT repository URLs.
    cloningPath: the directory that the repository will be cloned at.
    Optional Input:-
    threads_limit: The limit of working threads.
    username: Github username.
    token: Github token or password.
    """
    Q = queue.Queue()
    threads_state = []
    for URL in URLs:
        Q.put(URL)
    while Q.empty() is False:
        if (threading.active_count() < (threads_limit + 1)):
            t = threading.Thread(target=cloneRepo, args=(Q.get(), cloningPath,), kwargs={"username": username, "token": token})
            t.daemon = True
            t.start()
        else:
            time.sleep(0.5)
            threads_state.append(t)
    for _ in threads_state:
        _.join()

#==============================================================================

# Read/Write Paths
# ================
#---- Write
hoard_dir     = '/home/evan/Projects/Hoarded'
clone_path    = f'{hoard_dir}/clones'
user_org_path = f'{hoard_dir}/UserOrg'
archive_file  = f'{hoard_dir}/archive.json'

#---- Read
repo_inbox = f'{hoard_dir}/inbox.txt'
user_inbox = f'{hoard_dir}/users.txt'
org_inbox  = f'{hoard_dir}/orgs.txt'


KEYS = {repo_inbox: 'repos', org_inbox: 'orgs', user_inbox: 'users'}

# Read --> Write map
RW = {repo_inbox: clone_path,
      user_inbox: user_org_path,
      org_inbox:  user_org_path,}

# Formatting
# ==========
cur_date = str(dt.datetime.now().date())
start_tag_archive = f"# START {cur_date}\n"
end_tag_archive   = f"# END {cur_date}\n"




# Functions
# =========
def archive_urls(urls, key):
    with open(archive_file, 'r') as afile:
        archive_dict = json.load(afile)
    chive_set = set(archive_dict[key])
    chive_set.update(urls)
    chive_updated = list(chive_set)
    archive_dict[key] = chive_updated
    #archive_dict[key].extend(urls)

    with open(archive_file, 'w') as afile:
        json.dump(archive_dict, afile)
        #archive.write(start_tag_archive)
        #archive.write(urls)
        #archive.write(end_tag_archive)
    #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use


'''
def get_urls_from_file(txt=repo_inbox, archive=True):
    with open(txt, 'r+') as txt_urls:
        url_list = txt_urls.read().split('\n')[:-1]
        #url_list = url_string.split('\n')[:-1]
        if archive:
            #archive_urls(url_string)
            archive_urls(url_list, KEYS[txt])
            # clear inbox
            #txt_urls.truncate(0) # need '0' when using r+
        # Convert string to list
        url_list = url_string.split('\n')[:-1]
    return url_list
'''


def get_urls(txt=repo_inbox, archive=True, username=None, token=None):
    with open(txt, 'r+') as txt_urls:
        url_list = txt_urls.read().split('\n')[:-1]
        urls = list(url_list)
        if archive:
            archive_urls(url_list, KEYS[txt])
            # clear inbox
            #txt_urls.truncate(0) # need '0' when using r+
        if txt in [org_inbox, user_inbox]:
            urls = []
            uo_get_func = fromOrg if txt == org_inbox else fromUser
            for u in url_list:
                #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
                urls.extend(uo_get_func(u, username=username, token=token))
    return urls


urls = get_urls(org_inbox, username=USERNAME, token=TOKEN)
urls.extend(get_urls(user_inbox, username=USERNAME, token=TOKEN))
urls.extend(get_urls(repo_inbox, username=USERNAME, token=TOKEN))
#pprint(urls)

#urls = get_urls_from_file()
#pprint(urls)

cloneBulkRepos(urls, clone_path, threads_limit=8, username=USERNAME, token=TOKEN)

# URL origin
# ==========

#==== User

#==== Org

#==== txt
