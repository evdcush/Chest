# Modules
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

# Constants
HOME = os.environ["HOME"]
RESOURCE_PATH = f'{HOME}/Chest/Resources'
HOARD_PATH = f'{HOME}/Projects/Hoard'
HOARD_CLONES_PATH = f'{HOARD_PATH}/Clones'
HEADERS = {'Accept': '*/*'}
TIMEOUT = 3



class HoardJSON:
    """ Wraps pathing and json files related to hoard """
    keys = ['repos', 'users', 'orgs']
    hoard_inbox_path   = f'{RESOURCE_PATH}/hoard_inbox.json'
    hoard_archive_path = f'{RESOURCE_PATH}/hoard_archive.json'
    hoard_auth_path    = f'{RESOURCE_PATH}/gh_pub_token.json'

    def __init__(self):
        self.inbox   = self.read_json(self.hoard_inbox_path)
        self.archive = self.read_json(self.hoard_archive_path)
        self.auth    = self.read_json(self.hoard_auth_path)

    def update_archive(self, key, val, write=False):
        archive_set = set(self.archive[key])
        if isinstance(val, list):
            archive_set.update(val)
        else: # it's a single, string val url
            assert isinstance(val, str)
            archive_set.add(val)
        updated = list(archive_set)
        self.archive[key] = updated
        if write:
            self.write_json(self.hoard_archive_path, self.archive)

    def clear_inbox(self):
        cleared_inbox = {key: [] for key in self.keys}
        self.write_json(self.hoard_inbox_path, cleared_inbox)

    @staticmethod
    def write_json(file_path, obj):
        with open(file_path, 'w') as file:
            json.dump(obj, file)

    @staticmethod
    def read_json(file_path):
        with open(file_path, 'r') as file:
            obj = json.load(file)
        return obj






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
            resp = requests.get(API, headers=HEADERS, timeout=TIMEOUT).text
        else:
            resp = requests.get(API, headers=HEADERS, timeout=TIMEOUT, auth=(username, token)).text
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
            resp = requests.get(API, headers=HEADERS, timeout=TIMEOUT).text
        else:
            resp = requests.get(API, headers=HEADERS, timeout=TIMEOUT, auth=(username, token)).text
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




JSON_files = HoardJSON()
username, token = list(JSON_FILES.auth.items()).pop()



################## STOPPED HERE, UPDATE LATER
##########################################################
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
